from netgen.csg import *
from ngsolve import *
from ngsolve.internal import *


def MakeGeometry():
    geometry = CSGeometry()
    o_ext = (Sphere(Pnt(0,0,0), 2)).bc("outer")

    pml = Sphere(Pnt(0,0,0),1.5)

    o_minus = (Sphere(Pnt(0,0,0), 0.5)).maxh(0.2).mat("ominus").bc("interface")

    geometry.Add ((o_ext - pml).mat("air"))
    geometry.Add ((pml-o_minus).mat("air"))
    geometry.Add (o_minus)

    return geometry



ngmesh = MakeGeometry().GenerateMesh(maxh=0.2)
#ngmesh.Save("scatterer.vol")
mesh = Mesh(ngmesh)

ngsglobals.testout = "test.out"
SetPMLParameters(rad=1.5,alpha=1.7)
# curve elements for geometry approximation
mesh.Curve(5)

ngsglobals.msg_level = 5

#Vfull = HCurl(mesh, order=6)
Vext = HCurl(mesh, order=2, complex=True, definedon=mesh.Materials("air"), dirichlet="outer")
Vint = HCurl(mesh, order=2, complex=True, definedon=mesh.Materials("ominus"))
print(mesh.GetBoundaries())

V=FESpace([Vext,Vint])

#print("Vall:",Vall)
# u and v refer to trial and test-functions in the definition of forms below
uext,uint = V.TrialFunction()
vext,vint = V.TestFunction()

#Material parameters
k = 2  # wave number
mu2 = 1 # mu in oplus
mur = {"ominus" : mu2, "air" : 1 }
nu_coef = [ 1/(mur[mat]) for mat in mesh.GetMaterials() ]
nu = CoefficientFunction ([ 1/(mur[mat]) for mat in mesh.GetMaterials() ])

epsilonr = {"ominus" : 1, "air" : 1 }
epsilon_coef = [ epsilonr[mat] for mat in mesh.GetMaterials() ]
epsilon = CoefficientFunction (epsilon_coef)


nv = specialcf.normal(mesh.dim) # normal vector
Cross = lambda u,v: CoefficientFunction((u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])) # cross product


#Incident field
Ein= exp(k*1J*x)*CoefficientFunction ((0,1,0))
Einext=GridFunction (Vext,"gEin")
Einext.Set(Ein)
curlEin = k*1J*exp(k*1J*x)*CoefficientFunction((0,0,1))

#Sesquilinear form
a = BilinearForm(V, symmetric=False,flags={"printelmat":True})

a.components[0] += BFI("PML_curlcurledge",coef=nu)
a.components[0] += BFI("PML_massedge",coef=-k*k*epsilon)
a += SymbolicBFI(nu*curl(uint)*curl(vint) - k*k*epsilon*uint*vint)
a += SymbolicBFI((1./mu2)*curl(uint).Trace() * Cross(nv,vext.Trace()-vint.Trace()),BND,definedon=mesh.Boundaries("interface"))
a += SymbolicBFI( 1.e2 *  (uext.Trace() - uint.Trace()) * (vext.Trace() - vint.Trace()),BND,definedon=mesh.Boundaries("interface"))
#c = Preconditioner(a, type="bddcc")
#c = Preconditioner(a, type="multigrid", flags = { "smoother" : "block" } )

f = LinearForm(V)
f += SymbolicLFI( curlEin * Cross(nv,vint.Trace()),BND,definedon=mesh.Boundaries("interface"))
f += SymbolicLFI( -1.e2 *  Ein * (vext.Trace()-vint.Trace()),BND,definedon=mesh.Boundaries("interface"))

u = GridFunction(V)
viewoptions.clipping.enable = 1
viewoptions.clipping.ny = 0
viewoptions.clipping.nz = -1
visoptions.clipsolution = "scal"


Draw(u.components[0],mesh,"uext")
Draw(CoefficientFunction(u.components[1]),mesh,"uint")
Draw(Ein[1][0], mesh, "Ein")


Draw(u.components[1]+u.components[0]+Einext,mesh,"E")
Draw(u.components[0],mesh,"uext")
Draw(u.components[1],mesh,"uint")
#Draw(u.components[1],mesh,"Einner")

f.Assemble()
a.Assemble()


freedofs = BitArray(V.FreeDofs())

#solver = CGSolver(mat=a.mat, pre=c.mat)
u.vec.data = a.mat.Inverse(freedofs) * f.vec

Redraw()
