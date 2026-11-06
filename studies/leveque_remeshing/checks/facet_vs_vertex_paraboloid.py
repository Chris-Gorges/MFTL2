# Where does a facet-integral (Jibben-style) paraboloid fit put an edge midpoint
# on an inscribed icosphere, compared with a vertex-based fit and the sphere?
import numpy as np
def icosphere(n):
    t=(1+5**.5)/2
    V=[(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    F=[(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),(3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    V=[np.array(v)/np.linalg.norm(v) for v in V]
    for _ in range(n):
        cache={}; F2=[]
        def mid(a,b):
            k=(min(a,b),max(a,b))
            if k not in cache:
                m=V[a]+V[b]; V.append(m/np.linalg.norm(m)); cache[k]=len(V)-1
            return cache[k]
        for a,b,c in F:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a); F2+= [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        F=F2
    return np.array(V),np.array(F)
def frame(n):
    a=np.array([1.,0,0]) if abs(n[0])<0.9 else np.array([0,1.,0])
    e1=np.cross(n,a); e1/=np.linalg.norm(e1); e2=np.cross(n,e1); return np.array([e1,e2,n])
def wend(r,R): 
    q=r/R; return np.where(q<1,(1-q)**4*(1+4*q),0.0)
for ref in (2,3,4):
    P,F=icosphere(ref)
    a,b=F[0][0],F[0][1]; m=0.5*(P[a]+P[b]); n=m/np.linalg.norm(m); Fr=frame(n)
    h=np.linalg.norm(P[a]-P[b]); Rw=2.5*h
    rows=[];rhs=[];w=[]; vrows=[];vrhs=[];vw=[]
    for f in F:
        X=(P[f]-m)@Fr.T       # local coords
        c=X.mean(0)
        if np.linalg.norm(c)>Rw: continue
        nf=np.cross(X[1]-X[0],X[2]-X[0]); nf/=np.linalg.norm(nf)
        if nf[2]<0: nf=-nf
        pc=np.array([-(c@nf),nf[0],nf[1]])/(-nf[2])   # z = pc0 + pc1 x + pc2 y
        I=np.zeros(6)
        for k in range(3):
            xv,yv=X[k,0],X[k,1]; xn,yn=X[(k+1)%3,0],X[(k+1)%3,1]
            cr=xv*yn-xn*yv
            I+=[cr/2,(xv+xn)*cr/6,(yv+yn)*cr/6,(xv+xn)*(xv*xv+xn*xn)*(yn-yv)/12,
                (yn-yv)*(3*xv*xv*yv+xv*xv*yn+2*xv*xn*yv+2*xv*xn*yn+xn*xn*yv+3*xn*xn*yn)/24,
                (xv-xn)*(yv+yn)*(yv*yv+yn*yn)/12]
        if I[0]<0: I=-I
        rows.append(I); rhs.append(I[:3]@pc); w.append(wend(np.linalg.norm(c),Rw))
    for i,p in enumerate(P):
        X=(p-m)@Fr.T
        r=np.linalg.norm(X)
        if r>Rw: continue
        vrows.append([1,X[0],X[1],X[0]**2,X[0]*X[1],X[1]**2]); vrhs.append(X[2]); vw.append(wend(r,Rw))
    def solve(A,y,ww):
        A=np.array(A); y=np.array(y); s=np.sqrt(np.array(ww))
        return np.linalg.lstsq(A*s[:,None],y*s,rcond=None)[0]
    zj=solve(rows,rhs,w)[0]; zv=solve(vrows,vrhs,vw)[0]
    zs=1-np.linalg.norm(m)      # sphere above the chord midpoint
    print(f"refine {ref}: h={h:.4f}  sphere {zs:.3e}  vertex fit {zv:.3e} ({zv/zs:.3f})  facet-integral fit {zj:.3e} ({zj/zs:.3f})  [chord = 0]")
