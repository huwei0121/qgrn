# Full state sensitivities of the Stormer-Verlet discretization.
# The implicit systems are differentiated as exact linear solves. The native
# Neumann solve is configured to machine precision; parity is checked explicitly.
# All guarded-state rows at every real endpoint and imaginary midpoint enter GN.

function add_forcing!(dest,base,basis,sgn,E)
    n=size(base,1)
    @inbounds for j in eachindex(basis)
        b=sgn*basis[j]
        if b!=0
            off=(j-1)*E
            for col in 1:E,row in 1:n
                dest[row,off+col]+=b*base[row,col]
            end
        end
    end
    dest
end

function basis_at!(B,t,spline,free,temp)
    for q in axes(B,1)
        fill!(temp,0.)
        Juqbox.gradbcarrier2!(t,spline,q-1,temp)
        B[q,:].=temp[free]
    end
end

function gram_rows!(buffer,used,H,Dstate,state,weights,factor,E)
    n=size(state,1); d=size(buffer,2)
    grad=zeros(d)
    @inbounds for col in 1:E,row in 1:n
        w=factor*weights[row]
        w==0 && continue
        used+=1
        sw=sqrt(w)
        for j in 1:d
            deriv=Dstate[row,(j-1)*E+col]
            buffer[used,j]=sw*deriv
            grad[j]+=2w*state[row,col]*deriv
        end
        if used==size(buffer,1)
            mul!(H,transpose(buffer),buffer,2.,1.)
            used=0
        end
    end
    used,grad
end

function system_gn(c,x,node)
    p=c.p; n=size(p.Uinit,1); E=p.N; d=length(c.free); h=p.T/p.nsteps; q=h/2
    spline=Juqbox.bcparams(p.T,c.D1,p.Ncoupled,0,p.Cfreq,x)
    K0=zeros(n,n); S0=zeros(n,n); Km=zeros(n,n); Sm=zeros(n,n); K1=zeros(n,n); S1=zeros(n,n)
    Hbase=copy(p.Hconst)
    for j in 2:n
        Hbase[j,j]+=.01*node*10.0^(j-2)
    end
    u=copy(p.Uinit); v=zeros(n,E)
    du=zeros(n,E*d); dv=zeros(n,E*d)
    B0=zeros(2p.Ncoupled,d); Bm=similar(B0); B1=similar(B0); temp=zeros(c.nc)
    du0=similar(du); dv0=similar(du); dl=similar(du); dvm=similar(du)
    dk1=similar(du); dk2=similar(du); dl2=similar(du)
    buffer=zeros(2048,d); used=0; gram=zeros(d,d); gleak=zeros(d); leak=0.
    weights=diag(p.wmat); ident=Matrix{Float64}(I,n,n)
    ks!(K,S,t)=Juqbox.KS!(K,S,t,p.Hsym_ops,p.Hanti_ops,p.Hunc_ops,0,p.isSymm,spline,Hbase,p.Rfreq)
    for k in 1:p.nsteps
        t=(k-1)*h
        ks!(K0,S0,t);ks!(Km,Sm,t+q);ks!(K1,S1,t+h)
        basis_at!(B0,t,spline,c.free,temp);basis_at!(Bm,t+q,spline,c.free,temp);basis_at!(B1,t+h,spline,c.free,temp)
        Fm=lu(ident-q*Sm); F1=lu(ident-q*S1)
        u0=copy(u);v0=copy(v);copy!(du0,du);copy!(dv0,dv)
        l=Fm\(Km*u0+Sm*v0); vm=v0+q*l
        k1=S0*u0-K0*vm
        k2=F1\(S1*(u0+q*k1)-K1*vm)
        u=u0+q*(k1+k2)
        l2=Km*u+Sm*vm
        v=v0+q*(l+l2)
        mul!(dl,Km,du0);mul!(dl,Sm,dv0,1.,1.)
        for a in 1:p.Ncoupled
            add_forcing!(dl,p.Hsym_ops[a]*u0,view(Bm,2a-1,:),1.,E)
            add_forcing!(dl,p.Hanti_ops[a]*vm,view(Bm,2a,:),1.,E)
        end
        ldiv!(Fm,dl); @. dvm=dv0+q*dl
        mul!(dk1,S0,du0);mul!(dk1,K0,dvm,-1.,1.)
        for a in 1:p.Ncoupled
            add_forcing!(dk1,p.Hanti_ops[a]*u0,view(B0,2a,:),1.,E)
            add_forcing!(dk1,p.Hsym_ops[a]*vm,view(B0,2a-1,:),-1.,E)
        end
        @. dk2=du0+q*dk1
        mul!(du,S1,dk2);mul!(du,K1,dvm,-1.,1.)
        for a in 1:p.Ncoupled
            add_forcing!(du,p.Hanti_ops[a]*u,view(B1,2a,:),1.,E)
            add_forcing!(du,p.Hsym_ops[a]*vm,view(B1,2a-1,:),-1.,E)
        end
        ldiv!(F1,du);@. du=du0+q*(dk1+du)
        mul!(dl2,Km,du);mul!(dl2,Sm,dvm,1.,1.)
        for a in 1:p.Ncoupled
            add_forcing!(dl2,p.Hsym_ops[a]*u,view(Bm,2a-1,:),1.,E)
            add_forcing!(dl2,p.Hanti_ops[a]*vm,view(Bm,2a,:),1.,E)
        end
        @. dv=dv0+q*(dl+dl2)
        for (D,st,factor) in ((du0,u0,.5/p.nsteps),(du,u,.5/p.nsteps),(dvm,vm,1.0/p.nsteps))
            used,gg=gram_rows!(buffer,used,gram,D,st,weights,factor,E)
            gleak .+= gg
            leak+=factor*sum(weights.*sum(abs2,st,dims=2)[:])
        end
    end
    if used>0
        BB=view(buffer,1:used,:)
        mul!(gram,transpose(BB),BB,2.,1.)
    end
    U=complex.(u,-v); V=complex.(p.Utarget_r,p.Utarget_i)
    z=dot(V,U); fidelity=1-abs2(z)/E^2
    jac=zeros(ComplexF64,n*E,d); gate_grad=zeros(d); normgrad=zeros(d)
    for j in 1:d
        DU=complex.(du[:,(j-1)*E+1:j*E],-dv[:,(j-1)*E+1:j*E])
        dz=dot(V,DU)
        jac[:,j]=vec(DU-V*(dz/E))/sqrt(E)
        gate_grad[j]=-2real(conj(z)*dz)/E^2
        normgrad[j]=-2real(dot(U,DU))/E
    end
    gram .+= 2real.(jac'*jac)
    (;gram,g=gate_grad+gleak,gate_grad,leak,fidelity,
        norm_defect=1-sum(abs2,U)/E,norm_defect_gradient=normgrad,terminal=U)
end

function full_gn(c,x)
    t=time_ns();d=length(c.free);H=zeros(d,d);g=zeros(d)
    normdefects=Float64[]; forward_f=0.; max_state_error=0.
    for (node,weight) in zip(c.nodes,c.weights)
        a=system_gn(c,x,node)
        H .+= weight*a.gram;g .+= weight*a.g
        forward_f+=weight*(a.fidelity+a.leak)
        push!(normdefects,a.norm_defect)
    end
    tik=c.p.tik0/c.nc
    for j in 1:d
        H[j,j]+=2tik
    end
    g .+= 2tik*x[c.free];forward_f+=tik*sum(abs2,x)
    Dict("GN"=>H,"sensitivity_gradient"=>g,"sensitivity_objective"=>forward_f,
        "sv_norm_defects"=>normdefects,"gn_seconds"=>(time_ns()-t)*1e-9,
        "curvature_model"=>"full SV endpoint projective GN plus all path leakage GN and Tikhonov; exact native gradient includes SV norm-defect correction")
end
