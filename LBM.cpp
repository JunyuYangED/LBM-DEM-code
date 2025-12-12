#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>


using namespace std;

#define TX 100
#define TY 100
#define TZ 100
#define PX 6
#define PY 8
#define PZ 6
#define Q 19
#define QG 7
#define PXYZ (PX*PY*PZ)
#define NP 6
#define inter 1

const int ex[Q]={0, 1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0};
const int ey[Q]={0, 0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1};
const int ez[Q]={0, 0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1};
const int op[Q]={0, 2,  1,  4,  3,  6,  5, 10,  9,  8,  7,  14, 13,12, 11, 18, 17, 16, 15};
const double w[Q]={1.0/3,1.0/18,1.0/18,1.0/18,1.0/18,1.0/18,1.0/18,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36,1.0/36};
const double J[QG]={1.0/4,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8};

int i, j, k, l, n;

const double dx=1.0, dy=1.0, dz=1.0, dt=1.0;
double rhoi, rholA, rhogA, rholB, rhogB, uxi, uyi, uzi;
double niuA, niuB, xiA, xiB;
double rhol, rhog, vofl, vofg;

double ***rho, ***rho0, ***ux, ***uy, ***uz, ***ux0, ***uy0, ***uz0, ***p, ***p0;

double ***rhoA, ***rhoA0;
double ***phiA, ***psiA, ***TscA, ***pA;
double ****fA, mA[Q], ****fApost, MA[Q], sA[Q];
double ***fscABx, ***fscABy, ***fscABz, ***fscAAx, ***fscAAy, ***fscAAz, ***fscAx, ***fscAy, ***fscAz;
double ****sscA;
double GAB, ***GAA, rA;

double ***rhoB, ***rhoB0;
double ***phiB, ***psiB, ***TscB, ***pB;
double ****fB, mB[Q], ****fBpost, MB[Q], sB[Q];
double ***fscBAx, ***fscBAy, ***fscBAz, ***fscBBx, ***fscBBy, ***fscBBz, ***fscBx, ***fscBy, ***fscBz;
double ****sscB;
double ***GBB, rB;

double ***solid, ***bounce, ***solid0, ***dsolid;

int ***datafile, ***LG;

double rhoAin, rhoBin, uxin, uyin, uzin;
double rhoAout, rhoBout, uxout, uyout, uzout;
double lA, lB;
double gvx, gvy, gvz;
double kappa;
double betavof;
double theta;

double error;

double ***vof, ***dvofx, ***dvofy, ***dvofz, ***ddvof, ***vof0;
double ***drhox, ***drhoy, ***drhoz;
double vofc, vofw;
double sigma;

int destright, destleft, destfront, destback, destup, destdown;
double *send_right, *send_left, *send_front, *send_back, *send_up, *send_down;
double *recv_right, *recv_left, *recv_front, *recv_back, *recv_up, *recv_down;
double *sendfA_right, *sendfA_left, *sendfA_front, *sendfA_back, *sendfA_up, *sendfA_down;
double *recvfA_right, *recvfA_left, *recvfA_front, *recvfA_back, *recvfA_up, *recvfA_down;
double *sendfB_right, *sendfB_left, *sendfB_front, *sendfB_back, *sendfB_up, *sendfB_down;
double *recvfB_right, *recvfB_left, *recvfB_front, *recvfB_back, *recvfB_up, *recvfB_down;
double *sendmacro_right, *sendmacro_left, *sendmacro_front, *sendmacro_back, *sendmacro_up, *sendmacro_down;
double *recvmacro_right, *recvmacro_left, *recvmacro_front, *recvmacro_back, *recvmacro_up, *recvmacro_down;
const int tagrfA=1001, tagrfB=1002, taglfA=1003, taglfB=1004, tagffA=1005, tagffB=1006, tagbfA=1007, tagbfB=1008, tagufA=1009, tagufB=1010, tagdfA=1011, tagdfB=1012;
const int tagrm=2001, taglm=2002, tagfm=2003, tagbm=2004, tagum=2005, tagdm=2006;
const int Nmacro=2;//rhoA, rhoB

int mpisize, mpirank, rankx, ranky, rankz;
int lengthx, lengthy, lengthz;
int startx, starty, startz;
int endx, endy, endz;
int NX, NY, NZ;

double ***C, ***C0;
double ***CA, ***CB, ***CA0, ***CB0;
double ****Scst;
double ****g, ****gpost, mg[QG], Mg[QG], ****sg;
double DA, DB;
double ***kr, ***Ceq;
double kr0, Ceq0;
double ***D;
double MMA, MMB, Vm, H;
double Cinitial, Cinlet, Coutlet;
double errorC;
int upd;

double *sendg_right, *sendg_left, *sendg_front, *sendg_back, *sendg_up, *sendg_down;
double *recvg_right, *recvg_left, *recvg_front, *recvg_back, *recvg_up, *recvg_down;
const int tagrg=3001, taglg=3002, tagfg=3003, tagbg=3004, tagug=3005, tagdg=3006;
const int tagrC=4001, taglC=4002, tagfC=4003, tagbC=4004, taguC=4005, tagdC=4006;
const int tagrS=5001, taglS=5002, tagfS=5003, tagbS=5004, taguS=5005, tagdS=5006;

double ***T, ***T0;
double ****h, ****hpost, mh[QG], Mh[QG], ****sh;
double ****Sreac, ****Sconj;
double ***Sr, ***Sc;
double lamdaA, lamdaB, lamdaS, lamdaR, cpA, cpB, cpS, cpR, dH;
double Tinitial, Tinlet, Toutlet;
double ***lamda, ***rhocp, ***alpha;
double errorT;

double *sendh_right, *sendh_left, *sendh_front, *sendh_back, *sendh_up, *sendh_down;
double *recvh_right, *recvh_left, *recvh_front, *recvh_back, *recvh_up, *recvh_down;
const int tagrh=8001, taglh=8002, tagfh=8003, tagbh=8004, taguh=8005, tagdh=8006;
const int tagrT=9001, taglT=9002, tagfT=9003, tagbT=9004, taguT=9005, tagdT=9006;

double aveCs, aveTs, aveuxs, Hydrates, Surfaces, EffSurfaces;
int avens, avels;

const int tagrdvx=9101, tagldvx=9102, tagfdvx=9103, tagbdvx=9104, tagudvx=9105, tagddvx=9106;
const int tagrdvy=9201, tagldvy=9202, tagfdvy=9203, tagbdvy=9204, tagudvy=9205, tagddvy=9206;
const int tagrdvz=9301, tagldvz=9302, tagfdvz=9303, tagbdvz=9304, tagudvz=9305, tagddvz=9306;


const char* dataSp = "./OutData_PV";
const char* dataSp1= "/PV_";
const char* dataSpF1 = "/F_PV_";
const char* dataSp2= "/PCM_"; 
const char* dataF = "./File";
const char* dataF1= "/File_Para";
const char* dataF2= "/File_Info";
const char* dataF3= "/File_Ext";
int bigend;//for the computer system.

int NLag, NLagm, NLagn;
double rhopar;
double *mpar, *Ipar;
int pp, pl;
int nD;
double dtD;
double *parcx, *parcy, *parcz, *parr;
double *partheta, *parphi;
double paralpha;
double *parux, *paruy, *paruz;
double *parwx, *parwy, *parwz;
double **Lagx, **Lagy, **Lagz;
double **Lagtheta, **Lagphi;
double **Lagds;
int **Eulx, **Euly, **Eulz;
double *parFcx, *parFcy, *parFcz;
double *parTcx, *parTcy, *parTcz;
double *parFhx, *parFhy, *parFhz;
double *parThx, *parThy, *parThz;
double *parFrx, *parFry, *parFrz;
double *parTrx, *parTry, *parTrz;
double *parFfx, *parFfy, *parFfz;
double *parTfx, *parTfy, *parTfz;
double *parFtotalx, *parFtotaly, *parFtotalz;
double *parTtotalx, *parTtotaly, *parTtotalz;
double **parnwx, **parnwy, **parnwz;
double **parnx, **parny, **parnz;
double ****parsolid;
double ****parsolid0;
double ****paruwx, ****paruwy, ****paruwz;
double ***uwx, ***uwy, ***uwz;
double ***uwx0, ***uwy0, ***uwz0;
double ***Eulnwx, ***Eulnwy, ***Eulnwz;

int qq;
double deltapq;
double npqx, npqy, npqz;
double npqlen;
double cpqx, cpqy, cpqz;
double dpqx, dpqy, dpqz;
double upqx, upqy, upqz;
double upqlen;
double upqtx, upqty, upqtz;
double **deltapqtx, **deltapqty, **deltapqtz;
double **deltapqtx0, **deltapqty0, **deltapqtz0;
double deltapqtlen, deltapqtlen0;
double Fn, kn, gamman;
double Fttestx, Ftx, Fttesty, Fty, Fttestz, Ftz, kt, gammat;
double *Fpqx, *Fpqy, *Fpqz;
double Fttestlen;
double muf;
double *Tpqx, *Tpqy, *Tpqz;
double *deltapqtfrontx, *deltapqtfronty, *deltapqtfrontz;
double *deltapqtfrontx0, *deltapqtfronty0, *deltapqtfrontz0;
double *deltapqtbackx, *deltapqtbacky, *deltapqtbackz;
double *deltapqtbackx0, *deltapqtbacky0, *deltapqtbackz0;
double *deltapqttopx, *deltapqttopy, *deltapqttopz;
double *deltapqttopx0, *deltapqttopy0, *deltapqttopz0;
double *deltapqtbottomx, *deltapqtbottomy, *deltapqtbottomz;
double *deltapqtbottomx0, *deltapqtbottomy0, *deltapqtbottomz0;
double *deltapqtleftx, *deltapqtlefty, *deltapqtleftz;
double *deltapqtleftx0, *deltapqtlefty0, *deltapqtleftz0;
double *deltapqtrightx, *deltapqtrighty, *deltapqtrightz;
double *deltapqtrightx0, *deltapqtrighty0, *deltapqtrightz0;

double Frtestx, Frx, Frtesty, Fry, Frtestz, Frz, krpq, gammar;
double Frtestlen;
double mur;
double deltapqrlen, deltapqrlen0;
double upqrx, upqry, upqrz;

double *deltapqrfrontx, *deltapqrfronty, *deltapqrfrontz;
double *deltapqrfrontx0, *deltapqrfronty0, *deltapqrfrontz0;
double *deltapqrbackx, *deltapqrbacky, *deltapqrbackz;
double *deltapqrbackx0, *deltapqrbacky0, *deltapqrbackz0;
double *deltapqrtopx, *deltapqrtopy, *deltapqrtopz;
double *deltapqrtopx0, *deltapqrtopy0, *deltapqrtopz0;
double *deltapqrbottomx, *deltapqrbottomy, *deltapqrbottomz;
double *deltapqrbottomx0, *deltapqrbottomy0, *deltapqrbottomz0;
double *deltapqrleftx, *deltapqrlefty, *deltapqrleftz;
double *deltapqrleftx0, *deltapqrlefty0, *deltapqrleftz0;
double *deltapqrrightx, *deltapqrrighty, *deltapqrrightz;
double *deltapqrrightx0, *deltapqrrighty0, *deltapqrrightz0;

double parctotal;
double mlosstotal;
double maddtotal;
double ntotal;
double mlosstotaltemp;
double maddtotaltemp;
double ***msource;
double ***msourcetemp;


double gvxpar, gvypar, gvzpar;

void ProcessorID();
void Parameter();
double meq(int l,double rho,double p,double ux,double uy,double uz);
double mbeq(int l,double vof,double ux,double uy,double uz);
double feq(int l,double rho,double p,double ux,double uy,double uz);
double fbeq(int l,double vof,double ux,double uy,double uz);
void Phipsi();
void Initial();
void Initial2();
void Fsc();
void Fscb();
void Collision();
void Collisionb();
void Infosendrecvf();
void Boundary();
void Boundary2();
void Inlet();
void Outlet();
void InletZH();
void Streaming();
void Macro();
void Infosendrecvmacro();
void Getbounce();//inletoutlet
void Phisolid();//inletoutlet
void Phisolid2();
void Vof();//inletoutlet
void Input();
void Output(int m);
void Errorcheck();
void Outputpar(int m);

void Parameterg();
double geq(int l,double C,double ux,double uy,double uz);
double mgeq(int l,double C,double ux,double uy,double uz);
void Initialg();
void Dproperty();
void CSTsource();
void Collisiong();
void Infosendrecvg();
void Boundaryg();
void Boundaryg2();
void Inletg();
void Outletg();
void Streamingg();
void Macrog();
void InfosendrecvC();
void Solidevolution();
void Solidupdate();//inletoutlet
void Infosendrecvsolid();
void Getcacb();
void Errorcheckg();

void Parameterh();
double heq(int l,double T,double ux,double uy,double uz);
double mheq(int l,double C,double ux,double uy,double uz);
void Initialh();
void Tproperty();
void Heatsource();
void Conjsource();
void Collisionh();
void Infosendrecvh();
void Boundaryh();
void Inleth();
void Outleth();
void Streamingh();
void Macroh();
void InfosendrecvT();
void Errorcheckh();

void Statistics();
void Dsolidcom();

void Parameterpar();
void Initialpar();
double VCross(double ax, double ay, double az, double bx, double by, double bz, int l);
void Lagrangian();
void Eulerian();
void Infosendrecvparsolid();
void Parsolidupdate();
void Movement();
void Hydroforce();
void Capillaryforce();
void Infosendrecvuw();
void Infosendrecvdvof();
void Contactforce();
void Parboundaryfront();
void Parboundaryback();
void Parboundarytop();
void Parboundarybottom();
void Parboundaryleft();
void Parboundaryright();
void Inputpar();

void fOutputVTK3D();
void fOutputInfo();
int fBigEndian();//1
void fByteSwap(void *data, int len, int count);//2

int main(int argc,char**argv)
{
    double startwtime, endwtime;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    startwtime=MPI_Wtime();
    ProcessorID();
    Parameter();
    Parameterg();
    Parameterh();
    Parameterpar();
    Input();
    Inputpar();
    Initial();
    Initialh();
    Initialg();
    Initialpar();
    Lagrangian();
    Eulerian();
    Infosendrecvparsolid();
    Parsolidupdate();
    Parsolidupdate();
    Fscb();
    fOutputInfo();

    for(n=0;;n++)
    {
            Fscb();
            Fsc();
            Collisionb();
            Collision();
            Infosendrecvf();
            Boundary();
            Infosendrecvf();
            Streaming();
            Macro();
            Infosendrecvmacro();
            Phisolid();
            Infosendrecvmacro();

            Hydroforce();
            Capillaryforce();
            for(nD=0;nD<inter;nD++)
            {
                Contactforce();
                Movement();
                Lagrangian();
                Eulerian();
                Infosendrecvparsolid();
                Parsolidupdate();
            }
        

        if(n%1000==0)
        {
            Errorcheck();
        }

        if(n%10000==0) 
        {
            Outputpar(n);
            fOutputVTK3D();
        }
        if(n>1000000) break;
        if(isnan(error)) break;
        parctotal=0.0;
        for(pp=0;pp<NP;pp++)
        {
            parctotal=parctotal+parcx[pp]+parcy[pp]+parcz[pp];
        }
        if(isnan(parctotal)) break;
    }

    
    endwtime=MPI_Wtime();
    MPI_Finalize();
    return 0;
}

void ProcessorID()
{
    rankz=mpirank/(PX*PY);
    ranky=(mpirank-rankz*PX*PY)/PX;
    rankx=mpirank-rankz*PX*PY-ranky*PX;

    if (rankx<(TX%PX))
    {
        lengthx=TX/PX+1;
        startx=rankx*lengthx-1;
        endx=startx+lengthx+1;
    }
    else
    {
        lengthx=TX/PX;
        startx=rankx*lengthx+TX%PX-1;
        endx=startx+lengthx+1;
    }

    if (ranky<(TY%PY))
    {
        lengthy=TY/PY+1;
        starty=ranky*lengthy-1;
        endy=starty+lengthy+1;
    }
    else
    {
        lengthy=TY/PY;
        starty=ranky*lengthy+TY%PY-1;
        endy=starty+lengthy+1;
    }

    if (rankz<(TZ%PZ))
    {
        lengthz=TZ/PZ+1;
        startz=rankz*lengthz-1;
        endz=startz+lengthz+1;
    }
    else
    {
        lengthz=TZ/PZ;
        startz=rankz*lengthz+TZ%PZ-1;
        endz=startz+lengthz+1;
    }

    NX=lengthx+2;
    NY=lengthy+2;
    NZ=lengthz+2;
    destright=(rankx+1)%PX+ranky*PX+rankz*PX*PY;
    destleft=(rankx+PX-1)%PX+ranky*PX+rankz*PX*PY;
    destfront=rankx+(ranky+1)%PY*PX+rankz*PX*PY;
    destback=rankx+(ranky+PY-1)%PY*PX+rankz*PX*PY;
    destup=rankx+ranky*PX+(rankz+1)%PZ*PX*PY;
    destdown=rankx+ranky*PX+(rankz+PZ-1)%PZ*PX*PY;
}

void Parameter()
{
    rhol=1.0;
    rhog=1.0;
    vofl=1.0;
    vofg=0.0;
    vofc=0.5;
    vofw=5.0;
    sigma=0.001;

    betavof=12.0*sigma/vofw;
    kappa=3.0*sigma*vofw/2.0;
    theta=acos(-1.0)*30.0/180.0;
    lA=1.2;

    gvx=0.0;
    gvy=0.0;
    gvz=-1.96e-7*0.0;

    mlosstotal=0.0;
    maddtotal=0.0;
    ntotal=NX*NY*NZ;

    niuA=0.1;
    xiA=0.5;
    sA[0]=1.0;
	sA[1]=1.0;
	sA[2]=1.0;
	sA[3]=1.0;
	sA[4]=1.0/(3.0*niuA+0.5);
	sA[5]=1.0/(3.0*niuA+0.5);
	sA[6]=1.0/(3.0*niuA+0.5);
	sA[7]=1.0/(1.5*xiA+0.5);
	sA[8]=1.0/(3.0*niuA+0.5);
	sA[9]=1.0/(3.0*niuA+0.5);
	sA[10]=1.25;
	sA[11]=1.25;
	sA[12]=1.25;
	sA[13]=1.25;
	sA[14]=1.25;
	sA[15]=1.25;
	sA[16]=1.25;
	sA[17]=1.25;
	sA[18]=1.25;

    niuB=0.5;
    sB[0]=1.0;
	sB[1]=1.0/(3.0*niuB+0.5);
	sB[2]=1.0/(3.0*niuB+0.5);
	sB[3]=1.0/(3.0*niuB+0.5);
	sB[4]=1.0;
	sB[5]=1.0;
	sB[6]=1.0;
	sB[7]=1.0;
	sB[8]=1.0;
	sB[9]=1.0;
	sB[10]=1.0;
	sB[11]=1.0;
	sB[12]=1.0;
	sB[13]=1.0;
	sB[14]=1.0;
	sB[15]=1.0;
	sB[16]=1.0;
	sB[17]=1.0;
	sB[18]=1.0;

//double ***rho, ***rho0, ***ux, ***uy, ***uz, ***ux0, ***uy0, ***uz0;
    rho=new double**[NX];
    rho0=new double**[NX];
    ux=new double**[NX];
    uy=new double**[NX];
    uz=new double**[NX];
    ux0=new double**[NX];
    uy0=new double**[NX];
    uz0=new double**[NX];
    p=new double**[NX];
    p0=new double**[NX];
    msource=new double**[NX];
    msourcetemp=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rho[i]=new double*[NY];
        rho0[i]=new double*[NY];
        ux[i]=new double*[NY];
        uy[i]=new double*[NY];
        uz[i]=new double*[NY];
        ux0[i]=new double*[NY];
        uy0[i]=new double*[NY];
        uz0[i]=new double*[NY];
        p[i]=new double*[NY];
        p0[i]=new double*[NY];
        msource[i]=new double*[NY];
        msourcetemp[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rho[i][j]=new double[NZ];
            rho0[i][j]=new double[NZ];
            ux[i][j]=new double[NZ];
            uy[i][j]=new double[NZ];
            uz[i][j]=new double[NZ];
            ux0[i][j]=new double[NZ];
            uy0[i][j]=new double[NZ];
            uz0[i][j]=new double[NZ];
            p[i][j]=new double[NZ];
            p0[i][j]=new double[NZ];
            msource[i][j]=new double[NZ];
            msourcetemp[i][j]=new double[NZ];
        }

    rhoA=new double**[NX];
    rhoA0=new double**[NX];
    phiA=new double**[NX];
    psiA=new double**[NX];
    TscA=new double**[NX];
    pA=new double**[NX];
    fA=new double***[NX];
    fApost=new double***[NX];
    fscABx=new double**[NX];
    fscABy=new double**[NX];
    fscABz=new double**[NX];
    fscAAx=new double**[NX];
    fscAAy=new double**[NX];
    fscAAz=new double**[NX];
    fscAx=new double**[NX];
    fscAy=new double**[NX];
    fscAz=new double**[NX];
    sscA=new double***[NX];
    GAA=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rhoA[i]=new double*[NY];
        rhoA0[i]=new double*[NY];
        phiA[i]=new double*[NY];
        psiA[i]=new double*[NY];
        TscA[i]=new double*[NY];
        pA[i]=new double*[NY];
        fA[i]=new double**[NY];
        fApost[i]=new double**[NY];
        fscABx[i]=new double*[NY];
        fscABy[i]=new double*[NY];
        fscABz[i]=new double*[NY];
        fscAAx[i]=new double*[NY];
        fscAAy[i]=new double*[NY];
        fscAAz[i]=new double*[NY];
        fscAx[i]=new double*[NY];
        fscAy[i]=new double*[NY];
        fscAz[i]=new double*[NY];
        sscA[i]=new double**[NY];
        GAA[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rhoA[i][j]=new double[NZ];
            rhoA0[i][j]=new double[NZ];
            phiA[i][j]=new double[NZ];
            psiA[i][j]=new double[NZ];
            TscA[i][j]=new double[NZ];
            pA[i][j]=new double[NZ];
            fA[i][j]=new double*[NZ];
            fApost[i][j]=new double*[NZ];
            fscABx[i][j]=new double[NZ];
            fscABy[i][j]=new double[NZ];
            fscABz[i][j]=new double[NZ];
            fscAAx[i][j]=new double[NZ];
            fscAAy[i][j]=new double[NZ];
            fscAAz[i][j]=new double[NZ];
            fscAx[i][j]=new double[NZ];
            fscAy[i][j]=new double[NZ];
            fscAz[i][j]=new double[NZ];
            sscA[i][j]=new double*[NZ];
            GAA[i][j]=new double[NZ];
        }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
            for (k=0;k<NZ;k++)
            {
                fA[i][j][k]=new double[Q];
                fApost[i][j][k]=new double[Q];
                sscA[i][j][k]=new double[Q];
            }

    rhoB=new double**[NX];
    rhoB0=new double**[NX];
    phiB=new double**[NX];
    psiB=new double**[NX];
    TscB=new double**[NX];
    pB=new double**[NX];
    fB=new double***[NX];
    fBpost=new double***[NX];
    fscBAx=new double**[NX];
    fscBAy=new double**[NX];
    fscBAz=new double**[NX];
    fscBBx=new double**[NX];
    fscBBy=new double**[NX];
    fscBBz=new double**[NX];
    fscBx=new double**[NX];
    fscBy=new double**[NX];
    fscBz=new double**[NX];
    sscB=new double***[NX];
    GBB=new double**[NX];

    for (i=0;i<NX;i++)
    {
        rhoB[i]=new double*[NY];
        rhoB0[i]=new double*[NY];
        phiB[i]=new double*[NY];
        psiB[i]=new double*[NY];
        TscB[i]=new double*[NY];
        pB[i]=new double*[NY];
        fB[i]=new double**[NY];
        fBpost[i]=new double**[NY];
        fscBAx[i]=new double*[NY];
        fscBAy[i]=new double*[NY];
        fscBAz[i]=new double*[NY];
        fscBBx[i]=new double*[NY];
        fscBBy[i]=new double*[NY];
        fscBBz[i]=new double*[NY];
        fscBx[i]=new double*[NY];
        fscBy[i]=new double*[NY];
        fscBz[i]=new double*[NY];
        sscB[i]=new double**[NY];
        GBB[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            rhoB[i][j]=new double[NZ];
            rhoB0[i][j]=new double[NZ];
            phiB[i][j]=new double[NZ];
            psiB[i][j]=new double[NZ];
            TscB[i][j]=new double[NZ];
            pB[i][j]=new double[NZ];
            fB[i][j]=new double*[NZ];
            fBpost[i][j]=new double*[NZ];
            fscBAx[i][j]=new double[NZ];
            fscBAy[i][j]=new double[NZ];
            fscBAz[i][j]=new double[NZ];
            fscBBx[i][j]=new double[NZ];
            fscBBy[i][j]=new double[NZ];
            fscBBz[i][j]=new double[NZ];
            fscBx[i][j]=new double[NZ];
            fscBy[i][j]=new double[NZ];
            fscBz[i][j]=new double[NZ];
            sscB[i][j]=new double*[NZ];
            GBB[i][j]=new double[NZ];
        }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
            for (k=0;k<NZ;k++)
            {
                fB[i][j][k]=new double[Q];
                fBpost[i][j][k]=new double[Q];
                sscB[i][j][k]=new double[Q];
            }

    solid=new double**[NX];
    solid0=new double**[NX];
    dsolid=new double**[NX];
    bounce=new double**[NX];
    datafile=new int**[NX];
    LG=new int**[NX];
    vof=new double**[NX];
    dvofx=new double**[NX];
    dvofy=new double**[NX];
    dvofz=new double**[NX];
    ddvof=new double**[NX];
    drhox=new double**[NX];
    drhoy=new double**[NX];
    drhoz=new double**[NX];
    vof0=new double**[NX];

    for (i=0;i<NX;i++)
    {
        solid[i]=new double*[NY];
        solid0[i]=new double*[NY];
        dsolid[i]=new double*[NY];
        bounce[i]=new double*[NY];
        datafile[i]=new int*[NY];
        LG[i]=new int*[NY];
        vof[i]=new double*[NY];
        dvofx[i]=new double*[NY];
        dvofy[i]=new double*[NY];
        dvofz[i]=new double*[NY];
        ddvof[i]=new double*[NY];
        drhox[i]=new double*[NY];
        drhoy[i]=new double*[NY];
        drhoz[i]=new double*[NY];
        vof0[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
        for (j=0;j<NY;j++)
        {
            solid[i][j]=new double[NZ];
            solid0[i][j]=new double[NZ];
            dsolid[i][j]=new double[NZ];
            bounce[i][j]=new double[NZ];
            datafile[i][j]=new int[NZ];
            LG[i][j]=new int[NZ];
            vof[i][j]=new double[NZ];
            dvofx[i][j]=new double[NZ];
            dvofy[i][j]=new double[NZ];
            dvofz[i][j]=new double[NZ];
            ddvof[i][j]=new double[NZ];
            drhox[i][j]=new double[NZ];
            drhoy[i][j]=new double[NZ];
            drhoz[i][j]=new double[NZ];
            vof0[i][j]=new double[NZ];
        }

    send_right=new double[(NY-2)*(NZ-2)];
    send_left=new double[(NY-2)*(NZ-2)];
    send_front=new double[NX*(NZ-2)];
    send_back=new double[NX*(NZ-2)];
    send_up=new double[NX*NY];
    send_down=new double[NX*NY];
    recv_right=new double[(NY-2)*(NZ-2)];
    recv_left=new double[(NY-2)*(NZ-2)];
    recv_front=new double[NX*(NZ-2)];
    recv_back=new double[NX*(NZ-2)];
    recv_up=new double[NX*NY];
    recv_down=new double[NX*NY];

    sendfA_right=new double[(NY-2)*(NZ-2)*Q];
    sendfA_left=new double[(NY-2)*(NZ-2)*Q];
    sendfA_front=new double[NX*(NZ-2)*Q];
    sendfA_back=new double[NX*(NZ-2)*Q];
    sendfA_up=new double[NX*NY*Q];
    sendfA_down=new double[NX*NY*Q];
    recvfA_right=new double[(NY-2)*(NZ-2)*Q];
    recvfA_left=new double[(NY-2)*(NZ-2)*Q];
    recvfA_front=new double[NX*(NZ-2)*Q];
    recvfA_back=new double[NX*(NZ-2)*Q];
    recvfA_up=new double[NX*NY*Q];
    recvfA_down=new double[NX*NY*Q];

    sendfB_right=new double[(NY-2)*(NZ-2)*Q];
    sendfB_left=new double[(NY-2)*(NZ-2)*Q];
    sendfB_front=new double[NX*(NZ-2)*Q];
    sendfB_back=new double[NX*(NZ-2)*Q];
    sendfB_up=new double[NX*NY*Q];
    sendfB_down=new double[NX*NY*Q];
    recvfB_right=new double[(NY-2)*(NZ-2)*Q];
    recvfB_left=new double[(NY-2)*(NZ-2)*Q];
    recvfB_front=new double[NX*(NZ-2)*Q];
    recvfB_back=new double[NX*(NZ-2)*Q];
    recvfB_up=new double[NX*NY*Q];
    recvfB_down=new double[NX*NY*Q];

    sendmacro_right=new double[(NY-2)*(NZ-2)*Nmacro];
    sendmacro_left=new double[(NY-2)*(NZ-2)*Nmacro];
    sendmacro_front=new double[NX*(NZ-2)*Nmacro];
    sendmacro_back=new double[NX*(NZ-2)*Nmacro];
    sendmacro_up=new double[NX*NY*Nmacro];
    sendmacro_down=new double[NX*NY*Nmacro];
    recvmacro_right=new double[(NY-2)*(NZ-2)*Nmacro];
    recvmacro_left=new double[(NY-2)*(NZ-2)*Nmacro];
    recvmacro_front=new double[NX*(NZ-2)*Nmacro];
    recvmacro_back=new double[NX*(NZ-2)*Nmacro];
    recvmacro_up=new double[NX*NY*Nmacro];
    recvmacro_down=new double[NX*NY*Nmacro];
}

double meq(int l,double rho, double p, double ux,double uy,double uz)
{
	double meq;
		switch(l)
	    {
		case 0: {meq=p; break;}
		case 1: {meq=rho*ux/3.0; break;}
		case 2: {meq=rho*uy/3.0; break;}
		case 3: {meq=rho*uz/3.0; break;}
		case 4: {meq=rho*ux*uy/3.0; break;}
		case 5: {meq=rho*ux*uz/3.0; break;}
		case 6: {meq=rho*uy*uz/3.0; break;}
		case 7: {meq=p+rho*(ux*ux+uy*uy+uz*uz)/3.0; break;}
		case 8: {meq=rho*(ux*ux-uy*uy)/3.0; break;}
		case 9: {meq=rho*(ux*ux-uz*uz)/3.0; break;}
		case 10: {meq=rho*ux/9.0; break;}
		case 11: {meq=rho*ux/9.0; break;}
		case 12: {meq=rho*uy/9.0; break;}
		case 13: {meq=rho*uz/9.0; break;}
		case 14: {meq=rho*uy/9.0; break;}
		case 15: {meq=rho*uz/9.0; break;}
		case 16: {meq=(p+rho*ux*ux+rho*uy*uy-0.5*rho*uz*uz)/9.0; break;}
		case 17: {meq=(p+rho*ux*ux+rho*uz*uz-0.5*rho*uy*uy)/9.0; break;}
		case 18: {meq=(p+rho*uy*uy+rho*uz*uz-0.5*rho*ux*ux)/9.0; break;}
		default: meq=0.0;
		}
		return meq;
}

double feq(int l,double rho,double p,double ux,double uy,double uz)
{
	double eu,uv,feq;
	eu=ex[l]*ux+ey[l]*uy+ez[l]*uz;
	uv=ux*ux+uy*uy+uz*uz;
	feq=w[l]*rho*(3.0*eu+4.5*eu*eu-1.5*uv)/3.0+w[l]*p;
	return feq;
}

double mbeq(int l,double vof,double ux,double uy,double uz)
{
	double mbeq;
		switch(l)
	    {
		case 0: {mbeq=vof; break;}
		case 1: {mbeq=vof*ux; break;}
		case 2: {mbeq=vof*uy; break;}
		case 3: {mbeq=vof*uz; break;}
		case 4: {mbeq=0.0; break;}
		case 5: {mbeq=0.0; break;}
		case 6: {mbeq=0.0; break;}
		case 7: {mbeq=vof; break;}
		case 8: {mbeq=0.0; break;}
		case 9: {mbeq=0.0; break;}
		case 10: {mbeq=vof*ux/3.0; break;}
		case 11: {mbeq=vof*ux/3.0; break;}
		case 12: {mbeq=vof*uy/3.0; break;}
		case 13: {mbeq=vof*uz/3.0; break;}
		case 14: {mbeq=vof*uy/3.0; break;}
		case 15: {mbeq=vof*uz/3.0; break;}
		case 16: {mbeq=vof/9.0; break;}
		case 17: {mbeq=vof/9.0; break;}
		case 18: {mbeq=vof/9.0; break;}
		default: mbeq=0.0;
		}
		return mbeq;
}

double fbeq(int l,double vof,double ux,double uy,double uz)
{
	double eu,uv,fbeq;
	eu=ex[l]*ux+ey[l]*uy+ez[l]*uz;
	uv=ux*ux+uy*uy+uz*uz;
	fbeq=w[l]*vof*(1.0+3.0*eu);
	return fbeq;
}

void Initial()
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                p[i][j][k]=0.0;
                p0[i][j][k]=0.0;
                vof[i][j][k]=1.0;
                vof0[i][j][k]=1.0;
                ux[i][j][k]=0.0;
                uy[i][j][k]=0.0;
                uz[i][j][k]=0.0;
                ux0[i][j][k]=0.0;
                uy0[i][j][k]=0.0;
                uz0[i][j][k]=0.0;
                bounce[i][j][k]=0.0;

                fscAx[i][j][k]=0.0;
                fscAy[i][j][k]=0.0;
                fscAz[i][j][k]=0.0;
                fscBx[i][j][k]=0.0;
                fscBy[i][j][k]=0.0;
                fscBz[i][j][k]=0.0;


                if(datafile[i][j][k]==0)//gas
                {
                    vof[i][j][k]=vofg;
                    rho[i][j][k]=rhog;
                    p[i][j][k]=0.0;
                    solid[i][j][k]=0.0;
                }
                else if(datafile[i][j][k]==1)//liquid
                {
                    vof[i][j][k]=vofl;
                    rho[i][j][k]=rhol;
                    p[i][j][k]=0.0;
                    solid[i][j][k]=0.0;
                }
                else if(datafile[i][j][k]==2)//dissolved solid
                {
                    vof[i][j][k]=vofl;
                    rho[i][j][k]=rhol;
                    p[i][j][k]=0.0;
                    solid[i][j][k]=1.0;
                }
                else if(datafile[i][j][k]==3)//rock
                {
                    vof[i][j][k]=vofl;
                    rho[i][j][k]=rhol;
                    p[i][j][k]=0.0;
                    solid[i][j][k]=2.0;
                }

                for(l=0;l<Q;l++)
                {
                    fA[i][j][k][l]=feq(l,rho[i][j][k],p[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                    fB[i][j][k][l]=fbeq(l,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                }
                solid0[i][j][k]=solid[i][j][k];
            }
        Getbounce();
}

void Fsc()
{
    double niuvof;
    int ip,jp,kp;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                dvofx[i][j][k]=0.0;
                dvofy[i][j][k]=0.0;
                dvofz[i][j][k]=0.0;
                if(solid[i][j][k]==0.0)
                {
                    mB[1]=fB[i][j][k][1]-fB[i][j][k][2]+fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][11]-fB[i][j][k][12]+fB[i][j][k][13]-fB[i][j][k][14];
                    mB[2]=fB[i][j][k][3]-fB[i][j][k][4]+fB[i][j][k][7]+fB[i][j][k][8]-fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
                    mB[3]=fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
                     
                    dvofx[i][j][k]=3.0*fscBx[i][j][k]-3.0*sB[1]*(mB[1]-mbeq(1,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k])+0.5*sscB[i][j][k][1]);
                    dvofy[i][j][k]=3.0*fscBy[i][j][k]-3.0*sB[2]*(mB[2]-mbeq(2,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k])+0.5*sscB[i][j][k][2]);
                    dvofz[i][j][k]=3.0*fscBz[i][j][k]-3.0*sB[3]*(mB[3]-mbeq(3,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k])+0.5*sscB[i][j][k][3]);
                                        
                    /*dvofx[i][j][k]=0.0;
                    dvofy[i][j][k]=0.0;
                    dvofz[i][j][k]=0.0;
                    for(l=0;l<Q;l++)
                    {
                        dvofx[i][j][k]+=3.0*w[l]*ex[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                        dvofy[i][j][k]+=3.0*w[l]*ey[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                        dvofz[i][j][k]+=3.0*w[l]*ez[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                    }*/
                    
                    ddvof[i][j][k]=0.0;
                    for(l=0;l<Q;l++)
                    {
                        ddvof[i][j][k]+=2.0*3.0*w[l]*(vof[i+ex[l]][j+ey[l]][k+ez[l]]-vof[i][j][k]);
                    }
                    niuvof=4.0*betavof*(vof[i][j][k]-vofl)*(vof[i][j][k]-vofg)*(vof[i][j][k]-vofc)-kappa*ddvof[i][j][k];

                    fscAx[i][j][k]=niuvof*dvofx[i][j][k]+gvx*rho[i][j][k];
                    fscAy[i][j][k]=niuvof*dvofy[i][j][k]+gvy*rho[i][j][k];
                    fscAz[i][j][k]=niuvof*dvofz[i][j][k]+gvz*rho[i][j][k];

                    drhox[i][j][k]=(rhol-rhog)/(vofl-vofg)*dvofx[i][j][k];
                    drhoy[i][j][k]=(rhol-rhog)/(vofl-vofg)*dvofy[i][j][k];
                    drhoz[i][j][k]=(rhol-rhog)/(vofl-vofg)*dvofz[i][j][k];

                    sscA[i][j][k][0]=(drhox[i][j][k]*ux[i][j][k]+drhoy[i][j][k]*uy[i][j][k]+drhoz[i][j][k]*uz[i][j][k])/3.0;
	                sscA[i][j][k][1]=fscAx[i][j][k]/3.0;
	                sscA[i][j][k][2]=fscAy[i][j][k]/3.0;
	                sscA[i][j][k][3]=fscAz[i][j][k]/3.0;
	                sscA[i][j][k][4]=fscAx[i][j][k]*uy[i][j][k]/3.0+fscAy[i][j][k]*ux[i][j][k]/3.0+drhox[i][j][k]*uy[i][j][k]/9.0+drhoy[i][j][k]*ux[i][j][k]/9.0;
	                sscA[i][j][k][5]=fscAx[i][j][k]*uz[i][j][k]/3.0+fscAz[i][j][k]*ux[i][j][k]/3.0+drhox[i][j][k]*uz[i][j][k]/9.0+drhoz[i][j][k]*ux[i][j][k]/9.0;
	                sscA[i][j][k][6]=fscAy[i][j][k]*uz[i][j][k]/3.0+fscAz[i][j][k]*uy[i][j][k]/3.0+drhoy[i][j][k]*uz[i][j][k]/9.0+drhoz[i][j][k]*uy[i][j][k]/9.0;
	                sscA[i][j][k][7]=2.0/3.0*(fscAx[i][j][k]*ux[i][j][k]+fscAy[i][j][k]*uy[i][j][k]+fscAz[i][j][k]*uz[i][j][k])+5.0/9.0*(drhox[i][j][k]*ux[i][j][k]+drhoy[i][j][k]*uy[i][j][k]+drhoz[i][j][k]*uz[i][j][k]);
	                sscA[i][j][k][8]=fscAx[i][j][k]*ux[i][j][k]*2.0/3.0-fscAy[i][j][k]*uy[i][j][k]*2.0/3.0+drhox[i][j][k]*ux[i][j][k]*2.0/9.0-drhoy[i][j][k]*uy[i][j][k]*2.0/9.0;
	                sscA[i][j][k][9]=fscAx[i][j][k]*ux[i][j][k]*2.0/3.0-fscAz[i][j][k]*uz[i][j][k]*2.0/3.0+drhox[i][j][k]*ux[i][j][k]*2.0/9.0-drhoz[i][j][k]*uz[i][j][k]*2.0/9.0;
	                sscA[i][j][k][10]=uy[i][j][k]*uy[i][j][k]*drhox[i][j][k]/9.0+2.0*ux[i][j][k]*uy[i][j][k]*drhoy[i][j][k]/9.0+fscAx[i][j][k]/9.0;
	                sscA[i][j][k][11]=uz[i][j][k]*uz[i][j][k]*drhox[i][j][k]/9.0+2.0*ux[i][j][k]*uz[i][j][k]*drhoz[i][j][k]/9.0+fscAx[i][j][k]/9.0;
	                sscA[i][j][k][12]=ux[i][j][k]*ux[i][j][k]*drhoy[i][j][k]/9.0+2.0*ux[i][j][k]*uy[i][j][k]*drhox[i][j][k]/9.0+fscAy[i][j][k]/9.0;
	                sscA[i][j][k][13]=ux[i][j][k]*ux[i][j][k]*drhoz[i][j][k]/9.0+2.0*ux[i][j][k]*uz[i][j][k]*drhox[i][j][k]/9.0+fscAz[i][j][k]/9.0;
	                sscA[i][j][k][14]=uz[i][j][k]*uz[i][j][k]*drhoy[i][j][k]/9.0+2.0*uy[i][j][k]*uz[i][j][k]*drhoz[i][j][k]/9.0+fscAy[i][j][k]/9.0;
	                sscA[i][j][k][15]=uy[i][j][k]*uy[i][j][k]*drhoz[i][j][k]/9.0+2.0*uy[i][j][k]*uz[i][j][k]*drhoy[i][j][k]/9.0+fscAz[i][j][k]/9.0;
	                sscA[i][j][k][16]=ux[i][j][k]*fscAx[i][j][k]*2.0/9.0+ux[i][j][k]*drhox[i][j][k]/9.0+uy[i][j][k]*fscAy[i][j][k]*2.0/9.0+uy[i][j][k]*drhoy[i][j][k]/9.0-uz[i][j][k]*fscAz[i][j][k]/9.0;
	                sscA[i][j][k][17]=ux[i][j][k]*fscAx[i][j][k]*2.0/9.0+ux[i][j][k]*drhox[i][j][k]/9.0+uz[i][j][k]*fscAz[i][j][k]*2.0/9.0+uz[i][j][k]*drhoz[i][j][k]/9.0-uy[i][j][k]*fscAy[i][j][k]/9.0;
	                sscA[i][j][k][18]=uy[i][j][k]*fscAy[i][j][k]*2.0/9.0+uy[i][j][k]*drhoy[i][j][k]/9.0+uz[i][j][k]*fscAz[i][j][k]*2.0/9.0+uz[i][j][k]*drhoz[i][j][k]/9.0-ux[i][j][k]*fscAx[i][j][k]/9.0;
                }
            }
}

void Fscb()
{
    int ip,jp,kp;
    double ndvof;
    double phi;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    dvofx[i][j][k]=0.0;
                    dvofy[i][j][k]=0.0;
                    dvofz[i][j][k]=0.0;
                    for(l=0;l<Q;l++)
                    {
                        dvofx[i][j][k]+=3.0*w[l]*ex[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                        dvofy[i][j][k]+=3.0*w[l]*ey[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                        dvofz[i][j][k]+=3.0*w[l]*ez[l]*vof[i+ex[l]][j+ey[l]][k+ez[l]];
                    }
                    ndvof=sqrt(dvofx[i][j][k]*dvofx[i][j][k]+dvofy[i][j][k]*dvofy[i][j][k]+dvofz[i][j][k]*dvofz[i][j][k]);
                    ndvof=max(1e-12,ndvof);
                    phi=vofw/4.0*log(max(1e-20,vof[i][j][k])/max(1e-20,1.0-vof[i][j][k]));
                    /*fscBx[i][j][k]=(1.0-4.0*(vof[i][j][k]-vofc)*(vof[i][j][k]-vofc))/(3.0*vofw)*dvofx[i][j][k]/ndvof;
                    fscBy[i][j][k]=(1.0-4.0*(vof[i][j][k]-vofc)*(vof[i][j][k]-vofc))/(3.0*vofw)*dvofy[i][j][k]/ndvof;
                    fscBz[i][j][k]=(1.0-4.0*(vof[i][j][k]-vofc)*(vof[i][j][k]-vofc))/(3.0*vofw)*dvofz[i][j][k]/ndvof;*/

                    fscBx[i][j][k]=(1.0-tanh(2.0*phi/vofw)*tanh(2.0*phi/vofw))*dvofx[i][j][k]/ndvof/vofw/3.0;
                    fscBy[i][j][k]=(1.0-tanh(2.0*phi/vofw)*tanh(2.0*phi/vofw))*dvofy[i][j][k]/ndvof/vofw/3.0;
                    fscBz[i][j][k]=(1.0-tanh(2.0*phi/vofw)*tanh(2.0*phi/vofw))*dvofz[i][j][k]/ndvof/vofw/3.0;

                    msource[i][j][k]=msource[i][j][k]-0.0001*ndvof;

                    sscB[i][j][k][0]=0.0+msource[i][j][k];
                    sscB[i][j][k][1]=fscBx[i][j][k]+(vof[i][j][k]*ux[i][j][k]-vof0[i][j][k]*ux0[i][j][k]);
                    sscB[i][j][k][2]=fscBy[i][j][k]+(vof[i][j][k]*uy[i][j][k]-vof0[i][j][k]*uy0[i][j][k]);
                    sscB[i][j][k][3]=fscBz[i][j][k]+(vof[i][j][k]*uz[i][j][k]-vof0[i][j][k]*uz0[i][j][k]);
                    sscB[i][j][k][4]=0.0;
                    sscB[i][j][k][5]=0.0;
                    sscB[i][j][k][6]=0.0;
                    sscB[i][j][k][7]=0.0+msource[i][j][k];
                    sscB[i][j][k][8]=0.0;
                    sscB[i][j][k][9]=0.0;
                    sscB[i][j][k][10]=fscBx[i][j][k]/3.0+(vof[i][j][k]*ux[i][j][k]-vof0[i][j][k]*ux0[i][j][k])/3.0;
                    sscB[i][j][k][11]=fscBx[i][j][k]/3.0+(vof[i][j][k]*ux[i][j][k]-vof0[i][j][k]*ux0[i][j][k])/3.0;
                    sscB[i][j][k][12]=fscBy[i][j][k]/3.0+(vof[i][j][k]*uy[i][j][k]-vof0[i][j][k]*uy0[i][j][k])/3.0;
                    sscB[i][j][k][13]=fscBz[i][j][k]/3.0+(vof[i][j][k]*uz[i][j][k]-vof0[i][j][k]*uz0[i][j][k])/3.0;
                    sscB[i][j][k][14]=fscBy[i][j][k]/3.0+(vof[i][j][k]*uy[i][j][k]-vof0[i][j][k]*uy0[i][j][k])/3.0;
                    sscB[i][j][k][15]=fscBz[i][j][k]/3.0+(vof[i][j][k]*uz[i][j][k]-vof0[i][j][k]*uz0[i][j][k])/3.0;
                    sscB[i][j][k][16]=0.0+msource[i][j][k]/12.0;
                    sscB[i][j][k][17]=0.0+msource[i][j][k]/12.0;
                    sscB[i][j][k][18]=0.0+msource[i][j][k]/12.0;
                    
                }
                msource[i][j][k]=0.0;
            }
}

void Collision()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    mA[0]=fA[i][j][k][0]+fA[i][j][k][1]+fA[i][j][k][2]+fA[i][j][k][3]+fA[i][j][k][4]+fA[i][j][k][5]+fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18];
	                mA[1]=fA[i][j][k][1]-fA[i][j][k][2]+fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][11]-fA[i][j][k][12]+fA[i][j][k][13]-fA[i][j][k][14];
                    mA[2]=fA[i][j][k][3]-fA[i][j][k][4]+fA[i][j][k][7]+fA[i][j][k][8]-fA[i][j][k][9]-fA[i][j][k][10]+fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]-fA[i][j][k][18];
                    mA[3]=fA[i][j][k][5]-fA[i][j][k][6]+fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14]+fA[i][j][k][15]+fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
                    mA[4]=fA[i][j][k][7]-fA[i][j][k][8]-fA[i][j][k][9]+fA[i][j][k][10];
                    mA[5]=fA[i][j][k][11]-fA[i][j][k][12]-fA[i][j][k][13]+fA[i][j][k][14];
                    mA[6]=fA[i][j][k][15]-fA[i][j][k][16]-fA[i][j][k][17]+fA[i][j][k][18];
                    mA[7]=fA[i][j][k][1]+fA[i][j][k][2]+fA[i][j][k][3]+fA[i][j][k][4]+fA[i][j][k][5]+fA[i][j][k][6]+2.0*fA[i][j][k][7]+2.0*fA[i][j][k][8]+2.0*fA[i][j][k][9]+2.0*fA[i][j][k][10]+2.0*fA[i][j][k][11]+2.0*fA[i][j][k][12]+2.0*fA[i][j][k][13]+2.0*fA[i][j][k][14]+2.0*fA[i][j][k][15]+2.0*fA[i][j][k][16]+2.0*fA[i][j][k][17]+2.0*fA[i][j][k][18];
                    mA[8]=fA[i][j][k][1]+fA[i][j][k][2]-fA[i][j][k][3]-fA[i][j][k][4]+fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14]-fA[i][j][k][15]-fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
                    mA[9]=fA[i][j][k][1]+fA[i][j][k][2]-fA[i][j][k][5]-fA[i][j][k][6]+fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10]-fA[i][j][k][15]-fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
                    mA[10]=fA[i][j][k][7]-fA[i][j][k][8]+fA[i][j][k][9]-fA[i][j][k][10];
                    mA[11]=fA[i][j][k][11]-fA[i][j][k][12]+fA[i][j][k][13]-fA[i][j][k][14];
                    mA[12]=fA[i][j][k][7]+fA[i][j][k][8]-fA[i][j][k][9]-fA[i][j][k][10];
                    mA[13]=fA[i][j][k][11]+fA[i][j][k][12]-fA[i][j][k][13]-fA[i][j][k][14];
                    mA[14]=fA[i][j][k][15]-fA[i][j][k][16]+fA[i][j][k][17]-fA[i][j][k][18];
                    mA[15]=fA[i][j][k][15]+fA[i][j][k][16]-fA[i][j][k][17]-fA[i][j][k][18];
                    mA[16]=fA[i][j][k][7]+fA[i][j][k][8]+fA[i][j][k][9]+fA[i][j][k][10];
                    mA[17]=fA[i][j][k][11]+fA[i][j][k][12]+fA[i][j][k][13]+fA[i][j][k][14];
                    mA[18]=fA[i][j][k][15]+fA[i][j][k][16]+fA[i][j][k][17]+fA[i][j][k][18];

                    /*niuA=1.0*1.0/(vof[i][j][k]*1.0+(1.0-vof[i][j][k])*1.0);
                    if(vof[i][j][k]<0.5) niuA=1.22;
                    if(vof[i][j][k]>=0.5) niuA=1.22;
                    sA[4]=1.0/(3.0*niuA+0.5);
                    sA[5]=1.0/(3.0*niuA+0.5);
                    sA[6]=1.0/(3.0*niuA+0.5);
                    sA[8]=1.0/(3.0*niuA+0.5);
                    sA[9]=1.0/(3.0*niuA+0.5);*/

	                for(l=0;l<Q;l++)
	                {
		                MA[l]=mA[l]-sA[l]*(mA[l]-meq(l,rho[i][j][k],p[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]))+(1-sA[l]/2.0)*sscA[i][j][k][l];
	                }
	                fApost[i][j][k][0]=MA[0]-MA[7]+MA[16]+MA[17]+MA[18];
	                fApost[i][j][k][1]=MA[1]/2.0+MA[7]/6.0+MA[8]/6.0+MA[9]/6.0-MA[10]/2.0-MA[11]/2.0-MA[16]/2.0-MA[17]/2.0;
                    fApost[i][j][k][2]=-MA[1]/2.0+MA[7]/6.0+MA[8]/6.0+MA[9]/6.0+MA[10]/2.0+MA[11]/2.0-MA[16]/2.0-MA[17]/2.0;
                    fApost[i][j][k][3]=MA[2]/2.0+MA[7]/6.0-MA[8]/3.0+MA[9]/6.0-MA[12]/2.0-MA[14]/2.0-MA[16]/2.0-MA[18]/2.0;
                    fApost[i][j][k][4]=-MA[2]/2.0+MA[7]/6.0-MA[8]/3.0+MA[9]/6.0+MA[12]/2.0+MA[14]/2.0-MA[16]/2.0-MA[18]/2.0;
                    fApost[i][j][k][5]=MA[3]/2.0+MA[7]/6.0+MA[8]/6.0-MA[9]/3.0-MA[13]/2.0-MA[15]/2.0-MA[17]/2.0-MA[18]/2.0;
                    fApost[i][j][k][6]=MA[7]/6.0-MA[3]/2.0+MA[8]/6.0-MA[9]/3.0+MA[13]/2.0+MA[15]/2.0-MA[17]/2.0-MA[18]/2.0;
                    fApost[i][j][k][7]=MA[4]/4.0+MA[10]/4.0+MA[12]/4.0+MA[16]/4.0;
                    fApost[i][j][k][8]=MA[12]/4.0-MA[10]/4.0-MA[4]/4.0+MA[16]/4.0;
                    fApost[i][j][k][9]=MA[10]/4.0-MA[4]/4.0-MA[12]/4.0+MA[16]/4.0;
                    fApost[i][j][k][10]=MA[4]/4.0-MA[10]/4.0-MA[12]/4.0+MA[16]/4.0;
                    fApost[i][j][k][11]=MA[5]/4.0+MA[11]/4.0+MA[13]/4.0+MA[17]/4.0;
                    fApost[i][j][k][12]=MA[13]/4.0-MA[11]/4.0-MA[5]/4.0+MA[17]/4.0;
                    fApost[i][j][k][13]=MA[11]/4.0-MA[5]/4.0-MA[13]/4.0+MA[17]/4.0;
                    fApost[i][j][k][14]=MA[5]/4.0-MA[11]/4.0-MA[13]/4.0+MA[17]/4.0;
                    fApost[i][j][k][15]=MA[6]/4.0+MA[14]/4.0+MA[15]/4.0+MA[18]/4.0;
                    fApost[i][j][k][16]=MA[15]/4.0-MA[14]/4.0-MA[6]/4.0+MA[18]/4.0;
                    fApost[i][j][k][17]=MA[14]/4.0-MA[6]/4.0-MA[15]/4.0+MA[18]/4.0;
                    fApost[i][j][k][18]=MA[6]/4.0-MA[14]/4.0-MA[15]/4.0+MA[18]/4.0;
            }
        }
}

void Collisionb()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    mB[0]=fB[i][j][k][0]+fB[i][j][k][1]+fB[i][j][k][2]+fB[i][j][k][3]+fB[i][j][k][4]+fB[i][j][k][5]+fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18];
	                mB[1]=fB[i][j][k][1]-fB[i][j][k][2]+fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][11]-fB[i][j][k][12]+fB[i][j][k][13]-fB[i][j][k][14];
                    mB[2]=fB[i][j][k][3]-fB[i][j][k][4]+fB[i][j][k][7]+fB[i][j][k][8]-fB[i][j][k][9]-fB[i][j][k][10]+fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
                    mB[3]=fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14]+fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
                    mB[4]=fB[i][j][k][7]-fB[i][j][k][8]-fB[i][j][k][9]+fB[i][j][k][10];
                    mB[5]=fB[i][j][k][11]-fB[i][j][k][12]-fB[i][j][k][13]+fB[i][j][k][14];
                    mB[6]=fB[i][j][k][15]-fB[i][j][k][16]-fB[i][j][k][17]+fB[i][j][k][18];
                    mB[7]=fB[i][j][k][1]+fB[i][j][k][2]+fB[i][j][k][3]+fB[i][j][k][4]+fB[i][j][k][5]+fB[i][j][k][6]+2.0*fB[i][j][k][7]+2.0*fB[i][j][k][8]+2.0*fB[i][j][k][9]+2.0*fB[i][j][k][10]+2.0*fB[i][j][k][11]+2.0*fB[i][j][k][12]+2.0*fB[i][j][k][13]+2.0*fB[i][j][k][14]+2.0*fB[i][j][k][15]+2.0*fB[i][j][k][16]+2.0*fB[i][j][k][17]+2.0*fB[i][j][k][18];
                    mB[8]=fB[i][j][k][1]+fB[i][j][k][2]-fB[i][j][k][3]-fB[i][j][k][4]+fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14]-fB[i][j][k][15]-fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
                    mB[9]=fB[i][j][k][1]+fB[i][j][k][2]-fB[i][j][k][5]-fB[i][j][k][6]+fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10]-fB[i][j][k][15]-fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
                    mB[10]=fB[i][j][k][7]-fB[i][j][k][8]+fB[i][j][k][9]-fB[i][j][k][10];
                    mB[11]=fB[i][j][k][11]-fB[i][j][k][12]+fB[i][j][k][13]-fB[i][j][k][14];
                    mB[12]=fB[i][j][k][7]+fB[i][j][k][8]-fB[i][j][k][9]-fB[i][j][k][10];
                    mB[13]=fB[i][j][k][11]+fB[i][j][k][12]-fB[i][j][k][13]-fB[i][j][k][14];
                    mB[14]=fB[i][j][k][15]-fB[i][j][k][16]+fB[i][j][k][17]-fB[i][j][k][18];
                    mB[15]=fB[i][j][k][15]+fB[i][j][k][16]-fB[i][j][k][17]-fB[i][j][k][18];
                    mB[16]=fB[i][j][k][7]+fB[i][j][k][8]+fB[i][j][k][9]+fB[i][j][k][10];
                    mB[17]=fB[i][j][k][11]+fB[i][j][k][12]+fB[i][j][k][13]+fB[i][j][k][14];
                    mB[18]=fB[i][j][k][15]+fB[i][j][k][16]+fB[i][j][k][17]+fB[i][j][k][18];
	                for(l=0;l<Q;l++)
	                {
		                MB[l]=mB[l]-sB[l]*(mB[l]-mbeq(l,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]))+(1-sB[l]/2.0)*sscB[i][j][k][l];
	                }
	                fBpost[i][j][k][0]=MB[0]-MB[7]+MB[16]+MB[17]+MB[18];
	                fBpost[i][j][k][1]=MB[1]/2.0+MB[7]/6.0+MB[8]/6.0+MB[9]/6.0-MB[10]/2.0-MB[11]/2.0-MB[16]/2.0-MB[17]/2.0;
                    fBpost[i][j][k][2]=-MB[1]/2.0+MB[7]/6.0+MB[8]/6.0+MB[9]/6.0+MB[10]/2.0+MB[11]/2.0-MB[16]/2.0-MB[17]/2.0;
                    fBpost[i][j][k][3]=MB[2]/2.0+MB[7]/6.0-MB[8]/3.0+MB[9]/6.0-MB[12]/2.0-MB[14]/2.0-MB[16]/2.0-MB[18]/2.0;
                    fBpost[i][j][k][4]=-MB[2]/2.0+MB[7]/6.0-MB[8]/3.0+MB[9]/6.0+MB[12]/2.0+MB[14]/2.0-MB[16]/2.0-MB[18]/2.0;
                    fBpost[i][j][k][5]=MB[3]/2.0+MB[7]/6.0+MB[8]/6.0-MB[9]/3.0-MB[13]/2.0-MB[15]/2.0-MB[17]/2.0-MB[18]/2.0;
                    fBpost[i][j][k][6]=MB[7]/6.0-MB[3]/2.0+MB[8]/6.0-MB[9]/3.0+MB[13]/2.0+MB[15]/2.0-MB[17]/2.0-MB[18]/2.0;
                    fBpost[i][j][k][7]=MB[4]/4.0+MB[10]/4.0+MB[12]/4.0+MB[16]/4.0;
                    fBpost[i][j][k][8]=MB[12]/4.0-MB[10]/4.0-MB[4]/4.0+MB[16]/4.0;
                    fBpost[i][j][k][9]=MB[10]/4.0-MB[4]/4.0-MB[12]/4.0+MB[16]/4.0;
                    fBpost[i][j][k][10]=MB[4]/4.0-MB[10]/4.0-MB[12]/4.0+MB[16]/4.0;
                    fBpost[i][j][k][11]=MB[5]/4.0+MB[11]/4.0+MB[13]/4.0+MB[17]/4.0;
                    fBpost[i][j][k][12]=MB[13]/4.0-MB[11]/4.0-MB[5]/4.0+MB[17]/4.0;
                    fBpost[i][j][k][13]=MB[11]/4.0-MB[5]/4.0-MB[13]/4.0+MB[17]/4.0;
                    fBpost[i][j][k][14]=MB[5]/4.0-MB[11]/4.0-MB[13]/4.0+MB[17]/4.0;
                    fBpost[i][j][k][15]=MB[6]/4.0+MB[14]/4.0+MB[15]/4.0+MB[18]/4.0;
                    fBpost[i][j][k][16]=MB[15]/4.0-MB[14]/4.0-MB[6]/4.0+MB[18]/4.0;
                    fBpost[i][j][k][17]=MB[14]/4.0-MB[6]/4.0-MB[15]/4.0+MB[18]/4.0;
                    fBpost[i][j][k][18]=MB[6]/4.0-MB[14]/4.0-MB[15]/4.0+MB[18]/4.0;
            }
        }
}

void Infosendrecvf()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<Q;l++)
            {
                sendfA_right[l+(j-1+(k-1)*(NY-2))*Q]=fApost[NX-2][j][k][l];
                sendfA_left[l+(j-1+(k-1)*(NY-2))*Q]=fApost[1][j][k][l];
                sendfB_right[l+(j-1+(k-1)*(NY-2))*Q]=fBpost[NX-2][j][k][l];
                sendfB_left[l+(j-1+(k-1)*(NY-2))*Q]=fBpost[1][j][k][l];
            }
    MPI_Sendrecv(&sendfA_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,tagrfA,
                    &recvfA_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,tagrfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,taglfA,
                    &recvfA_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,taglfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,tagrfB,
                    &recvfB_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,tagrfB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_left[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destleft,taglfB,
                    &recvfB_right[0],(NY-2)*(NZ-2)*Q,MPI_DOUBLE,destright,taglfB,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
            for(l=0;l<Q;l++)
            {
                fApost[NX-1][j][k][l]=recvfA_right[l+(j-1+(k-1)*(NY-2))*Q];
                fApost[0][j][k][l]=recvfA_left[l+(j-1+(k-1)*(NY-2))*Q];
                fBpost[NX-1][j][k][l]=recvfB_right[l+(j-1+(k-1)*(NY-2))*Q];
                fBpost[0][j][k][l]=recvfB_left[l+(j-1+(k-1)*(NY-2))*Q];
            }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                sendfA_front[l+(i+(k-1)*(NX))*Q]=fApost[i][NY-2][k][l];
                sendfA_back[l+(i+(k-1)*(NX))*Q]=fApost[i][1][k][l];
                sendfB_front[l+(i+(k-1)*(NX))*Q]=fBpost[i][NY-2][k][l];
                sendfB_back[l+(i+(k-1)*(NX))*Q]=fBpost[i][1][k][l];
            }
    MPI_Sendrecv(&sendfA_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagffA,
                    &recvfA_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagffA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagbfA,
                    &recvfA_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagbfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagffB,
                    &recvfB_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagffB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_back[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destback,tagbfB,
                    &recvfB_front[0],(NX)*(NZ-2)*Q,MPI_DOUBLE,destfront,tagbfB,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                fApost[i][NY-1][k][l]=recvfA_front[l+(i+(k-1)*(NX))*Q];
                fApost[i][0][k][l]=recvfA_back[l+(i+(k-1)*(NX))*Q];
                fBpost[i][NY-1][k][l]=recvfB_front[l+(i+(k-1)*(NX))*Q];
                fBpost[i][0][k][l]=recvfB_back[l+(i+(k-1)*(NX))*Q];
            }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                sendfA_up[l+(i+j*(NX))*Q]=fApost[i][j][NZ-2][l];
                sendfA_down[l+(i+j*(NX))*Q]=fApost[i][j][1][l];
                sendfB_up[l+(i+j*(NX))*Q]=fBpost[i][j][NZ-2][l];
                sendfB_down[l+(i+j*(NX))*Q]=fBpost[i][j][1][l];
            }
    MPI_Sendrecv(&sendfA_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagufA,
                    &recvfA_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagufA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfA_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagdfA,
                    &recvfA_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagdfA,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagufB,
                    &recvfB_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagufB,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendfB_down[0],(NX)*(NY)*Q,MPI_DOUBLE,destdown,tagdfB,
                    &recvfB_up[0],(NX)*(NY)*Q,MPI_DOUBLE,destup,tagdfB,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
            for(l=0;l<Q;l++)
            {
                fApost[i][j][NZ-1][l]=recvfA_up[l+(i+j*(NX))*Q];
                fApost[i][j][0][l]=recvfA_down[l+(i+j*(NX))*Q];
                fBpost[i][j][NZ-1][l]=recvfB_up[l+(i+j*(NX))*Q];
                fBpost[i][j][0][l]=recvfB_down[l+(i+j*(NX))*Q];
            }
    MPI_Barrier(MPI_COMM_WORLD);
}

void Boundary()
{
    int ip,jp,kp;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(bounce[i][j][k]==1.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        ip=min(NX-1,max((i+ex[l]),0));
                        jp=min(NY-1,max((j+ey[l]),0));
                        kp=min(NZ-1,max((k+ez[l]),0));
                        fApost[i][j][k][l]=fApost[ip][jp][kp][op[l]]+2.0*w[op[l]]*rho[ip][jp][kp]*(ex[l]*uwx[i][j][k]+ey[l]*uwy[i][j][k]+ez[l]*uwz[i][j][k]);
                        fBpost[i][j][k][l]=fBpost[ip][jp][kp][op[l]]+2.0*w[op[l]]*vof[ip][jp][kp]*(ex[l]*uwx[i][j][k]+ey[l]*uwy[i][j][k]+ez[l]*uwz[i][j][k])*3.0;
                    }
                }

            }   
}

void Streaming()
{
    int iq, jq, kq;
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        iq=i-ex[l];
                        jq=j-ey[l];
                        kq=k-ez[l];
                        fA[i][j][k][l]=fApost[iq][jq][kq][l];
                        fB[i][j][k][l]=fBpost[iq][jq][kq][l];
                    }
                }
            }                
}

void Macro()
{
    double a;
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    p0[i][j][k]=p[i][j][k];
                    vof0[i][j][k]=vof[i][j][k];
                    ux0[i][j][k]=ux[i][j][k];
                    uy0[i][j][k]=uy[i][j][k];
                    uz0[i][j][k]=uz[i][j][k];
                }
            }

    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(solid[i][j][k]==0.0)
                {
                    p[i][j][k]=0.0;
                    vof[i][j][k]=0.0;
                    ux[i][j][k]=0.0;
                    uy[i][j][k]=0.0;
                    uz[i][j][k]=0.0;

                    for(l=0;l<Q;l++)
                    {
                        p[i][j][k]+=fA[i][j][k][l];
                        vof[i][j][k]+=fB[i][j][k][l];
                        ux[i][j][k]+=ex[l]*fA[i][j][k][l];
                        uy[i][j][k]+=ey[l]*fA[i][j][k][l];
                        uz[i][j][k]+=ez[l]*fA[i][j][k][l];
                    }

                    rho[i][j][k]=rhog+(vof[i][j][k]-vofg)/(vofl-vofg)*(rhol-rhog);
                    ux[i][j][k]=(3.0*ux[i][j][k]+fscAx[i][j][k]/2.0)/rho[i][j][k];
                    uy[i][j][k]=(3.0*uy[i][j][k]+fscAy[i][j][k]/2.0)/rho[i][j][k];
                    uz[i][j][k]=(3.0*uz[i][j][k]+fscAz[i][j][k]/2.0)/rho[i][j][k];
                    p[i][j][k]=p[i][j][k]+0.5/3.0*(rhol-rhog)/(vofl-vofg)*(ux[i][j][k]*dvofx[i][j][k]+uy[i][j][k]*dvofy[i][j][k]+uz[i][j][k]*dvofz[i][j][k]);
                }
            }

                

}

void Infosendrecvmacro()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro]=p[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro]=p[1][j][k];
            sendmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1]=vof[NX-2][j][k];
            sendmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1]=vof[1][j][k];
        }
    MPI_Sendrecv(&sendmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,tagrm,
                    &recvmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,tagrm,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_left[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destleft,taglm,
                    &recvmacro_right[0],(NY-2)*(NZ-2)*Nmacro,MPI_DOUBLE,destright,taglm,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            p[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro];
            p[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro];
            vof[NX-1][j][k]=recvmacro_right[(j-1+(k-1)*(NY-2))*Nmacro+1];
            vof[0][j][k]=recvmacro_left[(j-1+(k-1)*(NY-2))*Nmacro+1];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            sendmacro_front[(i+(k-1)*(NX))*Nmacro]=p[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro]=p[i][1][k];
            sendmacro_front[(i+(k-1)*(NX))*Nmacro+1]=vof[i][NY-2][k];
            sendmacro_back[(i+(k-1)*(NX))*Nmacro+1]=vof[i][1][k];
        }
    MPI_Sendrecv(&sendmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,tagfm,
                    &recvmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,tagfm,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_back[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destback,tagbm,
                    &recvmacro_front[0],(NX)*(NZ-2)*Nmacro,MPI_DOUBLE,destfront,tagbm,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            p[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro];
            p[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro];
            vof[i][NY-1][k]=recvmacro_front[(i+(k-1)*(NX))*Nmacro+1];
            vof[i][0][k]=recvmacro_back[(i+(k-1)*(NX))*Nmacro+1];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            sendmacro_up[(i+j*(NX))*Nmacro]=p[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro]=p[i][j][1];
            sendmacro_up[(i+j*(NX))*Nmacro+1]=vof[i][j][NZ-2];
            sendmacro_down[(i+j*(NX))*Nmacro+1]=vof[i][j][1];
        }
    MPI_Sendrecv(&sendmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,tagum,
                    &recvmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,tagum,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&sendmacro_down[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destdown,tagdm,
                    &recvmacro_up[0],(NX)*(NY)*Nmacro,MPI_DOUBLE,destup,tagdm,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            p[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro];
            p[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro];
            vof[i][j][NZ-1]=recvmacro_up[(i+j*(NX))*Nmacro+1];
            vof[i][j][0]=recvmacro_down[(i+j*(NX))*Nmacro+1];
        }

}

void Getbounce()
{
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                bounce[i][j][k]=0.0;
                if(solid[i][j][k]>0.0)
                {
                    for(l=0;l<Q;l++)
                    {
                        int ib, jb, kb;
                        ib=i+ex[l];
                        jb=j+ey[l];
                        kb=k+ez[l];
                        if(solid[ib][jb][kb]==0.0)
                        {
                            bounce[i][j][k]=1.0;
                        }
                    }
                }
            }

    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            send_right[j-1+(k-1)*(NY-2)]=bounce[NX-2][j][k];
            send_left[j-1+(k-1)*(NY-2)]=bounce[1][j][k];
        }
    MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,6001,
                    &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,6001,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,6002,
                    &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,6002,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            bounce[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
            bounce[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            send_front[i+(k-1)*(NX)]=bounce[i][NY-2][k];
            send_back[i+(k-1)*(NX)]=bounce[i][1][k];
        }
    MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,6003,
                    &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,6003,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,6004,
                    &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,6004,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            bounce[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
            bounce[i][0][k]=recv_back[i+(k-1)*(NX)];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            send_up[i+j*(NX)]=bounce[i][j][NZ-2];
            send_down[i+j*(NX)]=bounce[i][j][1];
        }
    MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,6005,
                    &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,6005,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,6006,
                    &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,6006,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            bounce[i][j][NZ-1]=recv_up[i+j*(NX)];
            bounce[i][j][0]=recv_down[i+j*(NX)];
        }

}

void Phisolid()
{
    double aveSA, aveSB, aven;
    double dsx, dsy, dsz;
    double dsxy, dsxyz;
    int ip, jp, kp;
    double a;
    int iq, jq, kq;

    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                if(bounce[i][j][k]==1.0)
                {
                    aveSA=0.0;
                    aven=0.0;
                    for(l=0;l<Q;l++)
                    {
                        ip=i+ex[l];
                        jp=j+ey[l];
                        kp=k+ez[l];
                        if(solid[ip][jp][kp]==0.0)
                        {
                            aveSA=aveSA+w[l]*vof[ip][jp][kp];
                            aven=aven+w[l];
                        }
                    }
                    if(aven>0.0)
                    {
                        aveSA=max(0.0,min(1.0,lA*aveSA/aven));
                    }
                    if(aven==0.0)
                    {
                        aveSA=0.5;
                    }

                    dsx=0.0;
                    dsy=0.0;
                    dsz=0.0;
                    for(l=0;l<Q;l++)
                    {
                        if(solid[i+ex[l]][j+ey[l]][k+ez[l]]>0.0)
                        {
                            dsx=dsx-3.0*w[l]*ex[l];
                            dsy=dsy-3.0*w[l]*ey[l];
                            dsz=dsz-3.0*w[l]*ez[l];
                        }
                    }

                    dsx=Eulnwx[i][j][k];
                    dsy=Eulnwy[i][j][k];
                    dsz=Eulnwz[i][j][k];

                    iq=0;
                    jq=0;
                    kq=0;

                    if(dsx>0.0)
                    {
                        if(dsy/dsx>=(-tan(acos(-1.0)/8.0))&&dsy/dsx<(tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(tan(acos(-1.0)/8.0))&&dsy/dsx<(tan(3.0*acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=1;
                                jq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=1;
                                jq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=1;
                                jq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(-3.0*tan(acos(-1.0)/8.0))&&dsy/dsx<(-tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=1;
                                jq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=1;
                                jq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=1;
                                jq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(3.0*tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx<(-3.0*tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        } 
                    }

                    if(dsx<0.0)
                    {
                        if(dsy/dsx>=(-tan(acos(-1.0)/8.0))&&dsy/dsx<(tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(tan(acos(-1.0)/8.0))&&dsy/dsx<(tan(3.0*acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=-1;
                                jq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                jq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                jq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(-3.0*tan(acos(-1.0)/8.0))&&dsy/dsx<(-tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                iq=-1;
                                jq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                jq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                iq=-1;
                                jq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx>=(3.0*tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy/dsx<(-3.0*tan(acos(-1.0)/8.0)))
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        } 
                    }

                    if(dsx==0.0)
                    {
                        if(dsy>0.0)
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy<0.0)
                        {
                            dsxy=sqrt(dsx*dsx+dsy*dsy);
                            if(dsz/dsxy>=(-tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(acos(-1.0)/8.0))) 
                            {
                                jq=-1;
                            }
                            if(dsz/dsxy>=(-tan(3.0*acos(-1.0)/8.0))&&dsz/dsxy<(-tan(acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=-1;
                            }
                            if(dsz/dsxy>=(tan(acos(-1.0)/8.0))&&dsz/dsxy<(tan(3.0*acos(-1.0)/8.0)))
                            {
                                jq=-1;
                                kq=1;
                            }
                            if(dsz/dsxy>=(tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=1;
                            }
                            if(dsz/dsxy<(-tan(3.0*acos(-1.0)/8.0)))
                            {
                                kq=-1;
                            }
                        }

                        if(dsy==0.0)
                        {
                            if(dsz>0.0) 
                            {
                                kq=1;
                            }
                            if(dsz<0.0)
                            {
                                kq=-1;
                            }
                        }
                    }

                    if(solid[i][j][k]==2.0) theta=acos(-1.0);
                    if(solid[i][j][k]<=1.0) theta=acos(-1.0)/180.0*30.0;

                    a=-0.5*sqrt(iq*iq+jq*jq+kq*kq)*sqrt(2*betavof/kappa)*cos(theta);
                    if (a!=0.0&&solid[i+iq][j+jq][k+kq]==0.0)
                    {
                        vof[i][j][k]=1.0/a*(1.0+a-sqrt((1.0+a)*(1.0+a)-4.0*a*vof[i+iq][j+jq][k+kq]))-vof[i+iq][j+jq][k+kq];
                        if(vof[i][j][k]>1.0||vof[i][j][k]<0.0)
                        {
                            vof[i][j][k]=1.0/a*(1.0+a+sqrt((1.0+a)*(1.0+a)-4.0*a*vof[i+iq][j+jq][k+kq]))-vof[i+iq][j+jq][k+kq];
                        }
                        if(vof[i][j][k]>1.0||vof[i][j][k]<0.0)
                        {
                            vof[i][j][k]=aveSA;
                        }
                    }
                    if (a==0.0&&solid[i+iq][j+jq][k+kq]<1.0)
                    {
                        vof[i][j][k]=vof[i+iq][j+jq][k+kq];
                    }
                    if (solid[i+iq][j+jq][k+kq]>=1.0)
                    {
                        vof[i][j][k]=aveSA;
                    }
                }
            }
}

void Input()
{
    FILE *fp;
    char filenamedata[20];
    sprintf(filenamedata,"./data/%s%.4d%s","data",mpirank,".dat");
    fp=fopen(filenamedata,"r");

    if(NULL==fp)
    {
        cout<<mpirank<<"data open error"<<endl;
    }

    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                fscanf(fp,"%i\n",&datafile[i][j][k]);
            }
    fclose(fp);
}

void Inputpar()
{
    FILE *fp2;
    char filenamedata2[20];
    sprintf(filenamedata2,"./particle.dat");
    fp2=fopen(filenamedata2,"r");

    if(NULL==fp2)
    {
        cout<<mpirank<<"data open error"<<endl;
    }

    for(pp=0;pp<NP;pp++)
    {
        fscanf(fp2,"%lf %lf %lf\n",&parcx[pp], &parcy[pp], &parcz[pp]);
    }
    fclose(fp2);
}


void Errorcheck()
{
    double temp1, temp2;
    temp1=0.0;
    temp2=0.0;
    for(i=1;i<NX-1;i++)
		for(j=1;j<NY-1;j++)
			for(k=1;k<NZ-1;k++)
			{
				if(solid[i][j][k]==0.0)
				{
					//temp1+=((ux[i][j][k]-ux0[i][j][k])*(ux[i][j][k]-ux0[i][j][k])+(uy[i][j][k]-uy0[i][j][k])*(uy[i][j][k]-uy0[i][j][k])+(uz[i][j][k]-uz0[i][j][k])*(uz[i][j][k]-uz0[i][j][k]));
  		            //temp2+=(ux[i][j][k]*ux[i][j][k]+uy[i][j][k]*uy[i][j][k]+uz[i][j][k]*uz[i][j][k]);

                    temp1+=(vof0[i][j][k]-vof[i][j][k])*(vof0[i][j][k]-vof[i][j][k]);
                    temp2+=vof0[i][j][k]*vof0[i][j][k];
				}
			}
			temp1=sqrt(temp1);
			temp2=sqrt(temp2);
			temp1=temp1/(temp2+1E-30);
    MPI_Allreduce(&temp1,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if(mpirank==0)
    {
        /*cout<<"The"<<n<<"th computation result:"<<endl;
        cout<<"The max relative error of rhoA is:"<<setiosflags(ios::scientific)<<error<<endl;*/
        //cout<<n<<" "<<parcz[1]<<endl;
    }
}

void Output(int m)
{
    ostringstream command;
    command<<"mkdir -p "<<"./LBM_"<<m;
    system(command.str().c_str());

    ostringstream name;
	name<<"./LBM_"<<m<<"/Proc_"<<mpirank<<".dat";
	ofstream out(name.str().c_str());

	for(k=1;k<NZ-1;k++)
		for(j=1;j<NY-1;j++)
			for(i=1;i<NX-1;i++)
			{
                out<<ux[i][j][k]<<" "<<uy[i][j][k]<<" "<<uz[i][j][k]<<" "<<p[i][j][k]<<" "<<rho[i][j][k]<<" "<<vof[i][j][k]<<" "<<dsolid[i][j][k]<<" "<<solid[i][j][k]<<" "<<C[i][j][k]<<" "<<CA[i][j][k]<<" "<<CB[i][j][k]<<" "<<T[i][j][k]<<endl;
                //out<<ux[i][j][k]<<" "<<uy[i][j][k]<<" "<<uz[i][j][k]<<" "<<rhoA[i][j][k]<<" "<<rhoB[i][j][k]<<" "<<vof[i][j][k]<<" "<<p[i][j][k]<<" "<<solid[i][j][k]<<endl;
			}
}

void Outputpar(int m)
{
    if(m==0)
    {
        ostringstream command;
        command<<"mkdir -p "<<"./particle";
        system(command.str().c_str());
    }
    
    ostringstream name;
	name << "./particle/point" << std::setw(8) << std::setfill('0') << m << ".csv";
	ofstream out(name.str().c_str());

    out<<"x,y,z,r"<<endl;

    for(pp=0;pp<NP;pp++)
    {
        out<<parcx[pp]<<","<<parcy[pp]<<","<<parcz[pp]<<","<<parr[pp]<<endl;
    }

    out.close();
}

void Infosendrecvsolid()
{
    MPI_Status status;
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            send_right[j-1+(k-1)*(NY-2)]=solid[NX-2][j][k];
            send_left[j-1+(k-1)*(NY-2)]=solid[1][j][k];
        }
    MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                    &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                    &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(j=1;j<NY-1;j++)
        {
            solid[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
            solid[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
        }

    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            send_front[i+(k-1)*(NX)]=solid[i][NY-2][k];
            send_back[i+(k-1)*(NX)]=solid[i][1][k];
        }
    MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                    &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                    &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
    for(k=1;k<NZ-1;k++)
        for(i=0;i<NX;i++)
        {
            solid[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
            solid[i][0][k]=recv_back[i+(k-1)*(NX)];
        }

    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            send_up[i+j*(NX)]=solid[i][j][NZ-2];
            send_down[i+j*(NX)]=solid[i][j][1];
        }
    MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                    &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
    MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                    &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
    for(j=0;j<NY;j++)
        for(i=0;i<NX;i++)
        {
            solid[i][j][NZ-1]=recv_up[i+j*(NX)];
            solid[i][j][0]=recv_down[i+j*(NX)];
        }
}

double VCross(double ax, double ay, double az, double bx, double by, double bz, int l)
{
    double ans;
    ans=0;
    if(l==0)
    {
        ans=ay*bz-az*by;
    }
    if(l==1)
    {
        ans=az*bx-ax*bz;
    }
    if(l==2)
    {
        ans=ax*by-ay*bx;
    }
    return ans;
}

void Parameterpar()
{
    NLagm=31;
    NLagn=2*(NLagm-1);
    NLag=NLagm*NLagn;
    paralpha=acos(-1.0)/(NLagm-1.0);
    dtD=dt/inter;
    rhopar=1.0*rhol;

    kn=1.0;
    gamman=0.0;

    kt=1.0;
    gammat=0.0;
    muf=0.2;

    krpq=0.02;
    gammar=0.0;
    mur=0.0;

    parcx=new double[NP];
    parcy=new double[NP];
    parcz=new double[NP];
    parr=new double[NP];
    mpar=new double[NP];
    Ipar=new double[NP];
    partheta=new double[NP];
    parphi=new double[NP];
    parux=new double[NP];
    paruy=new double[NP];
    paruz=new double[NP];
    parwx=new double[NP];
    parwy=new double[NP];
    parwz=new double[NP];
    parFtotalx=new double[NP];
    parFtotaly=new double[NP];
    parFtotalz=new double[NP];
    parTtotalx=new double[NP];
    parTtotaly=new double[NP];
    parTtotalz=new double[NP];
    parFhx=new double[NP];
    parFhy=new double[NP];
    parFhz=new double[NP];
    parThx=new double[NP];
    parThy=new double[NP];
    parThz=new double[NP];
    parFrx=new double[NP];
    parFry=new double[NP];
    parFrz=new double[NP];
    parTrx=new double[NP];
    parTry=new double[NP];
    parTrz=new double[NP];
    parFfx=new double[NP];
    parFfy=new double[NP];
    parFfz=new double[NP];
    parTfx=new double[NP];
    parTfy=new double[NP];
    parTfz=new double[NP];
    parFcx=new double[NP];
    parFcy=new double[NP];
    parFcz=new double[NP];
    parTcx=new double[NP];
    parTcy=new double[NP];
    parTcz=new double[NP];

    Fpqx=new double[NP];
    Fpqy=new double[NP];
    Fpqz=new double[NP];
    Tpqx=new double[NP];
    Tpqy=new double[NP];
    Tpqz=new double[NP];
    deltapqtfrontx=new double[NP];
    deltapqtfronty=new double[NP];
    deltapqtfrontz=new double[NP];
    deltapqtfrontx0=new double[NP];
    deltapqtfronty0=new double[NP];
    deltapqtfrontz0=new double[NP];
    deltapqtbackx=new double[NP];
    deltapqtbacky=new double[NP];
    deltapqtbackz=new double[NP];
    deltapqtbackx0=new double[NP];
    deltapqtbacky0=new double[NP];
    deltapqtbackz0=new double[NP];
    deltapqttopx=new double[NP];
    deltapqttopy=new double[NP];
    deltapqttopz=new double[NP];
    deltapqttopx0=new double[NP];
    deltapqttopy0=new double[NP];
    deltapqttopz0=new double[NP];
    deltapqtbottomx=new double[NP];
    deltapqtbottomy=new double[NP];
    deltapqtbottomz=new double[NP];
    deltapqtbottomx0=new double[NP];
    deltapqtbottomy0=new double[NP];
    deltapqtbottomz0=new double[NP];
    deltapqtleftx=new double[NP];
    deltapqtlefty=new double[NP];
    deltapqtleftz=new double[NP];
    deltapqtleftx0=new double[NP];
    deltapqtlefty0=new double[NP];
    deltapqtleftz0=new double[NP];
    deltapqtrightx=new double[NP];
    deltapqtrighty=new double[NP];
    deltapqtrightz=new double[NP];
    deltapqtrightx0=new double[NP];
    deltapqtrighty0=new double[NP];
    deltapqtrightz0=new double[NP];

    deltapqrfrontx=new double[NP];
    deltapqrfronty=new double[NP];
    deltapqrfrontz=new double[NP];
    deltapqrfrontx0=new double[NP];
    deltapqrfronty0=new double[NP];
    deltapqrfrontz0=new double[NP];
    deltapqrbackx=new double[NP];
    deltapqrbacky=new double[NP];
    deltapqrbackz=new double[NP];
    deltapqrbackx0=new double[NP];
    deltapqrbacky0=new double[NP];
    deltapqrbackz0=new double[NP];
    deltapqrtopx=new double[NP];
    deltapqrtopy=new double[NP];
    deltapqrtopz=new double[NP];
    deltapqrtopx0=new double[NP];
    deltapqrtopy0=new double[NP];
    deltapqrtopz0=new double[NP];
    deltapqrbottomx=new double[NP];
    deltapqrbottomy=new double[NP];
    deltapqrbottomz=new double[NP];
    deltapqrbottomx0=new double[NP];
    deltapqrbottomy0=new double[NP];
    deltapqrbottomz0=new double[NP];
    deltapqrleftx=new double[NP];
    deltapqrlefty=new double[NP];
    deltapqrleftz=new double[NP];
    deltapqrleftx0=new double[NP];
    deltapqrlefty0=new double[NP];
    deltapqrleftz0=new double[NP];
    deltapqrrightx=new double[NP];
    deltapqrrighty=new double[NP];
    deltapqrrightz=new double[NP];
    deltapqrrightx0=new double[NP];
    deltapqrrighty0=new double[NP];
    deltapqrrightz0=new double[NP];

    deltapqtx=new double*[NP];
    deltapqty=new double*[NP];
    deltapqtz=new double*[NP];
    deltapqtx0=new double*[NP];
    deltapqty0=new double*[NP];
    deltapqtz0=new double*[NP];

    for(pp=0;pp<NP;pp++)
    {
        deltapqtx[pp]=new double[NP];
        deltapqty[pp]=new double[NP];
        deltapqtz[pp]=new double[NP];
        deltapqtx0[pp]=new double[NP];
        deltapqty0[pp]=new double[NP];
        deltapqtz0[pp]=new double[NP];
    }

    Lagx=new double*[NP];
    Lagy=new double*[NP];
    Lagz=new double*[NP];
    Lagtheta=new double*[NP];
    Lagphi=new double*[NP];
    Lagds=new double*[NP];
    parnwx=new double*[NP];
    parnwy=new double*[NP];
    parnwz=new double*[NP];
    parnx=new double*[NP];
    parny=new double*[NP];
    parnz=new double*[NP];
    Eulx=new int*[NP];
    Euly=new int*[NP];
    Eulz=new int*[NP];

    for (pp=0;pp<NP;pp++)
    {
        Lagx[pp]=new double[NLag];
        Lagy[pp]=new double[NLag];
        Lagz[pp]=new double[NLag];
        Lagtheta[pp]=new double[NLag];
        Lagphi[pp]=new double[NLag];
        Lagds[pp]=new double[NLag];
        parnwx[pp]=new double[NLag];
        parnwy[pp]=new double[NLag];
        parnwz[pp]=new double[NLag];
        parnx[pp]=new double[NLag];
        parny[pp]=new double[NLag];
        parnz[pp]=new double[NLag];
        Eulx[pp]=new int[NLag];
        Euly[pp]=new int[NLag];
        Eulz[pp]=new int[NLag];
    }

    parsolid=new double***[NP];
    parsolid0=new double***[NP];
    paruwx=new double***[NP];
    paruwy=new double***[NP];
    paruwz=new double***[NP];
    for(pp=0;pp<NP;pp++)
    {
        parsolid[pp]=new double**[NX];
        parsolid0[pp]=new double**[NX];
        paruwx[pp]=new double**[NX];
        paruwy[pp]=new double**[NX];
        paruwz[pp]=new double**[NX];
    }
    for (pp=0;pp<NP;pp++)
    for (i=0;i<NX;i++)
    {
        parsolid[pp][i]=new double*[NY];
        parsolid0[pp][i]=new double*[NY];
        paruwx[pp][i]=new double*[NY];
        paruwy[pp][i]=new double*[NY];
        paruwz[pp][i]=new double*[NY];
    }

    for (pp=0;pp<NP;pp++)
    for (i=0;i<NX;i++)
    for (j=0;j<NY;j++)
    {
        parsolid[pp][i][j]=new double[NZ];
        parsolid0[pp][i][j]=new double[NZ];
        paruwx[pp][i][j]=new double[NZ];
        paruwy[pp][i][j]=new double[NZ];
        paruwz[pp][i][j]=new double[NZ];
    }

    uwx=new double**[NX];
    uwy=new double**[NX];
    uwz=new double**[NX];
    uwx0=new double**[NX];
    uwy0=new double**[NX];
    uwz0=new double**[NX];
    Eulnwx=new double**[NX];
    Eulnwy=new double**[NX];
    Eulnwz=new double**[NX];

    for (i=0;i<NX;i++)
    {
        uwx[i]=new double*[NY];
        uwy[i]=new double*[NY];
        uwz[i]=new double*[NY];
        uwx0[i]=new double*[NY];
        uwy0[i]=new double*[NY];
        uwz0[i]=new double*[NY];
        Eulnwx[i]=new double*[NY];
        Eulnwy[i]=new double*[NY];
        Eulnwz[i]=new double*[NY];
    }

    for (i=0;i<NX;i++)
    for (j=0;j<NY;j++)
    {
        uwx[i][j]=new double[NZ];
        uwy[i][j]=new double[NZ];
        uwz[i][j]=new double[NZ];
        uwx0[i][j]=new double[NZ];
        uwy0[i][j]=new double[NZ];
        uwz0[i][j]=new double[NZ];
        Eulnwx[i][j]=new double[NZ];
        Eulnwy[i][j]=new double[NZ];
        Eulnwz[i][j]=new double[NZ];
    }

}

void Initialpar()
{
    

    for(pp=0;pp<NP;pp++)
    {
        //parcx[pp]=TX/2.0;
        //parcy[pp]=TY/2.0;
        //parcz[pp]=217.0;
        parr[pp]=10.0;
        mpar[pp]=4.0/3.0*acos(-1.0)*parr[pp]*parr[pp]*parr[pp]*rhopar;
        Ipar[pp]=2.0/5.0*mpar[pp]*parr[pp]*parr[pp];
        partheta[pp]=0.0;
        parphi[pp]=0.0;
        parux[pp]=0.0;
        paruy[pp]=0.0;
        paruz[pp]=0.0;
        parwx[pp]=0.0;
        parwy[pp]=0.0;
        parwz[pp]=0.0;

        parFtotalx[pp]=0.0;
        parFtotaly[pp]=0.0;
        parFtotalz[pp]=0.0;
        parTtotalx[pp]=0.0;
        parTtotaly[pp]=0.0;
        parTtotalz[pp]=0.0;
        parFhx[pp]=0.0;
        parFhy[pp]=0.0;
        parFhz[pp]=0.0;
        parThx[pp]=0.0;
        parThy[pp]=0.0;
        parThz[pp]=0.0;
        parFrx[pp]=0.0;
        parFry[pp]=0.0;
        parFrz[pp]=0.0;
        parTrx[pp]=0.0;
        parTry[pp]=0.0;
        parTrz[pp]=0.0;
        parFfx[pp]=0.0;
        parFfy[pp]=0.0;
        parFfz[pp]=0.0;
        parTfx[pp]=0.0;
        parTfy[pp]=0.0;
        parTfz[pp]=0.0;
        parFcx[pp]=0.0;
        parFcy[pp]=0.0;
        parFcz[pp]=0.0;
        parTcx[pp]=0.0;
        parTcy[pp]=0.0;
        parTcz[pp]=0.0;
        Fpqx[pp]=0.0;
        Fpqy[pp]=0.0;
        Fpqz[pp]=0.0;
        Tpqx[pp]=0.0;
        Tpqy[pp]=0.0;
        Tpqz[pp]=0.0;

        deltapqtfrontx[pp]=0.0;
        deltapqtfronty[pp]=0.0;
        deltapqtfrontz[pp]=0.0;
        deltapqtfrontx0[pp]=0.0;
        deltapqtfronty0[pp]=0.0;
        deltapqtfrontz0[pp]=0.0;
        deltapqtbackx[pp]=0.0;
        deltapqtbacky[pp]=0.0;
        deltapqtbackz[pp]=0.0;
        deltapqtbackx0[pp]=0.0;
        deltapqtbacky0[pp]=0.0;
        deltapqtbackz0[pp]=0.0;
        deltapqttopx[pp]=0.0;
        deltapqttopy[pp]=0.0;
        deltapqttopz[pp]=0.0;
        deltapqttopx0[pp]=0.0;
        deltapqttopy0[pp]=0.0;
        deltapqttopz0[pp]=0.0;
        deltapqtbottomx[pp]=0.0;
        deltapqtbottomy[pp]=0.0;
        deltapqtbottomz[pp]=0.0;
        deltapqtbottomx0[pp]=0.0;
        deltapqtbottomy0[pp]=0.0;
        deltapqtbottomz0[pp]=0.0;
        deltapqtleftx[pp]=0.0;
        deltapqtlefty[pp]=0.0;
        deltapqtleftz[pp]=0.0;
        deltapqtleftx0[pp]=0.0;
        deltapqtlefty0[pp]=0.0;
        deltapqtleftz0[pp]=0.0;
        deltapqtrightx[pp]=0.0;
        deltapqtrighty[pp]=0.0;
        deltapqtrightz[pp]=0.0;
        deltapqtrightx0[pp]=0.0;
        deltapqtrighty0[pp]=0.0;
        deltapqtrightz0[pp]=0.0;


        deltapqrfrontx[pp]=0.0;
        deltapqrfronty[pp]=0.0;
        deltapqrfrontz[pp]=0.0;
        deltapqrfrontx0[pp]=0.0;
        deltapqrfronty0[pp]=0.0;
        deltapqrfrontz0[pp]=0.0;
        deltapqrbackx[pp]=0.0;
        deltapqrbacky[pp]=0.0;
        deltapqrbackz[pp]=0.0;
        deltapqrbackx0[pp]=0.0;
        deltapqrbacky0[pp]=0.0;
        deltapqrbackz0[pp]=0.0;
        deltapqrtopx[pp]=0.0;
        deltapqrtopy[pp]=0.0;
        deltapqrtopz[pp]=0.0;
        deltapqrtopx0[pp]=0.0;
        deltapqrtopy0[pp]=0.0;
        deltapqrtopz0[pp]=0.0;
        deltapqrbottomx[pp]=0.0;
        deltapqrbottomy[pp]=0.0;
        deltapqrbottomz[pp]=0.0;
        deltapqrbottomx0[pp]=0.0;
        deltapqrbottomy0[pp]=0.0;
        deltapqrbottomz0[pp]=0.0;
        deltapqrleftx[pp]=0.0;
        deltapqrlefty[pp]=0.0;
        deltapqrleftz[pp]=0.0;
        deltapqrleftx0[pp]=0.0;
        deltapqrlefty0[pp]=0.0;
        deltapqrleftz0[pp]=0.0;
        deltapqrrightx[pp]=0.0;
        deltapqrrighty[pp]=0.0;
        deltapqrrightz[pp]=0.0;
        deltapqrrightx0[pp]=0.0;
        deltapqrrighty0[pp]=0.0;
        deltapqrrightz0[pp]=0.0;

        for (pl=0;pl<NLag;pl++)
        {
            Lagx[pp][pl]=0.0;
            Lagy[pp][pl]=0.0;
            Lagz[pp][pl]=0.0;
            Lagtheta[pp][pl]=0.0;
            Lagphi[pp][pl]=0.0;
            Lagds[pp][pl]=0.0;
            parnwx[pp][pl]=0.0;
            parnwy[pp][pl]=0.0;
            parnwz[pp][pl]=0.0;
            parnx[pp][pl]=0.0;
            parny[pp][pl]=0.0;
            parnz[pp][pl]=0.0;
            Eulx[pp][pl]=0;
            Euly[pp][pl]=0;
            Eulz[pp][pl]=0;
        }

        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    parsolid[pp][i][j][k]=0.0;
                    paruwx[pp][i][j][k]=0.0;
                    paruwy[pp][i][j][k]=0.0;
                    paruwz[pp][i][j][k]=0.0;
                }

        for(qq=0;qq<NP;qq++)
        {
            deltapqtx[pp][qq]=0.0;
            deltapqty[pp][qq]=0.0;
            deltapqtz[pp][qq]=0.0;
            deltapqtx0[pp][qq]=0.0;
            deltapqty0[pp][qq]=0.0;
            deltapqtz0[pp][qq]=0.0;
        }
    }
    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                uwx[i][j][k]=0.0;
                uwy[i][j][k]=0.0;
                uwz[i][j][k]=0.0;
                uwx0[i][j][k]=0.0;
                uwy0[i][j][k]=0.0;
                uwz0[i][j][k]=0.0;
                Eulnwx[i][j][k]=0.0;
                Eulnwy[i][j][k]=0.0;
                Eulnwz[i][j][k]=0.0;
            }
        
}

void Lagrangian()
{
    for (pp=0;pp<NP;pp++)
    {
        pl=0;
        for(i=0;i<NLagm;i++)
        {
            for(j=0;j<NLagn;j++)
            {
                Lagtheta[pp][pl]=i*paralpha;
                Lagphi[pp][pl]=j*paralpha;
                pl=pl+1;
            }
        }

        for(pl=0;pl<NLag;pl++)
        {
            Lagx[pp][pl]=parcx[pp]+parr[pp]*sin(Lagtheta[pp][pl])*cos(Lagphi[pp][pl]);
            Lagy[pp][pl]=parcy[pp]+parr[pp]*sin(Lagtheta[pp][pl])*sin(Lagphi[pp][pl]);
            Lagz[pp][pl]=parcz[pp]+parr[pp]*cos(Lagtheta[pp][pl]);
            parnwx[pp][pl]=sin(Lagtheta[pp][pl])*cos(Lagphi[pp][pl]);
            parnwy[pp][pl]=sin(Lagtheta[pp][pl])*sin(Lagphi[pp][pl]);
            parnwz[pp][pl]=cos(Lagtheta[pp][pl]);
            Lagds[pp][pl]=parr[pp]*paralpha*parr[pp]*sin(Lagtheta[pp][pl])*paralpha;

            if(Lagx[pp][pl]<0.0)
            {
                Lagx[pp][pl]=Lagx[pp][pl]+TX;
            }
            if(Lagx[pp][pl]>=TX)
            {
                Lagx[pp][pl]=Lagx[pp][pl]-TX;
            }
            if(Lagy[pp][pl]<0.0)
            {
                Lagy[pp][pl]=Lagy[pp][pl]+TY;
            }
            if(Lagy[pp][pl]>=TY)
            {
                Lagy[pp][pl]=Lagy[pp][pl]-TY;
            }
            if(Lagz[pp][pl]<0.0)
            {
                Lagz[pp][pl]=Lagz[pp][pl]+TZ;
            }
            if(Lagz[pp][pl]>=TZ)
            {
                Lagz[pp][pl]=Lagz[pp][pl]-TZ;
            }
        }
    }
}

void Eulerian()
{
    double dx, dy, dz;
    for(pp=0;pp<NP;pp++)
    {
        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    parsolid0[pp][i][j][k]=parsolid[pp][i][j][k];
                    parsolid[pp][i][j][k]=0.0;
                    paruwx[pp][i][j][k]=0.0;
                    paruwy[pp][i][j][k]=0.0;
                    paruwz[pp][i][j][k]=0.0;

                    dx=min(abs(i+startx+0.5-parcx[pp]),min(abs(i+startx+0.5+TX-parcx[pp]),abs(i+startx+0.5-TX-parcx[pp])));
                    dy=min(abs(j+starty+0.5-parcy[pp]),min(abs(j+starty+0.5+TY-parcy[pp]),abs(j+starty+0.5-TY-parcy[pp])));
                    dz=min(abs(k+startz+0.5-parcz[pp]),min(abs(k+startz+0.5+TZ-parcz[pp]),abs(k+startz+0.5-TZ-parcz[pp])));

                    if(dx==abs(i+startx+0.5-parcx[pp])) dx=i+startx+0.5-parcx[pp];
                    if(dx==abs(i+startx+0.5+TX-parcx[pp])) dx=i+startx+0.5+TX-parcx[pp];
                    if(dx==abs(i+startx+0.5-TX-parcx[pp])) dx=i+startx+0.5-TX-parcx[pp];

                    if(dy==abs(j+starty+0.5-parcy[pp])) dy=j+starty+0.5-parcy[pp];
                    if(dy==abs(j+starty+0.5+TY-parcy[pp])) dy=j+starty+0.5+TY-parcy[pp];
                    if(dy==abs(j+starty+0.5-TY-parcy[pp])) dy=j+starty+0.5-TY-parcy[pp];

                    if(dz==abs(k+startz+0.5-parcz[pp])) dz=k+startz+0.5-parcz[pp];
                    if(dz==abs(k+startz+0.5+TZ-parcz[pp])) dz=k+startz+0.5+TZ-parcz[pp];
                    if(dz==abs(k+startz+0.5-TZ-parcz[pp])) dz=k+startz+0.5-TZ-parcz[pp];

                    if((dx)*(dx)+(dy)*(dy)+(dz)*(dz)<(parr[pp]-1.0)*(parr[pp]-1.0))
                    {
                        parsolid[pp][i][j][k]=2.0;
                        paruwx[pp][i][j][k]=parux[pp]+VCross( parwx[pp], parwy[pp], parwz[pp], dx, dy, dz,0);
                        paruwy[pp][i][j][k]=paruy[pp]+VCross( parwx[pp], parwy[pp], parwz[pp], dx, dy, dz,1);
                        paruwz[pp][i][j][k]=paruz[pp]+VCross( parwx[pp], parwy[pp], parwz[pp], dx, dy, dz,2);
                    }
                }

        for(pl=0;pl<NLag;pl++)
        {
            if(Lagx[pp][pl]>=(1.0+startx)&&Lagx[pp][pl]<(NX-1.0+startx))
            {
                if(Lagy[pp][pl]>=(1.0+starty)&&Lagy[pp][pl]<(NY-1.0+starty))
                {
                    if(Lagz[pp][pl]>=(1.0+startz)&&Lagz[pp][pl]<(NZ-1.0+startz))
                    {
                        Eulx[pp][pl]=int(Lagx[pp][pl]-startx);
                        Euly[pp][pl]=int(Lagy[pp][pl]-starty);
                        Eulz[pp][pl]=int(Lagz[pp][pl]-startz);
                        parsolid[pp][Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]]=0.0;
                    }
                }
            }
        }
    }

    for(i=1;i<NX-1;i++)
        for(j=1;j<NY-1;j++)
            for(k=1;k<NZ-1;k++)
            {
                Eulnwx[i][j][k]=0.0;
                Eulnwy[i][j][k]=0.0;
                Eulnwz[i][j][k]=0.0;
                for(pp=0;pp<NP;pp++)
                {
                    if(parsolid[pp][i][j][k]==2.0)
                    {
                        dx=min(abs(i+startx+0.5-parcx[pp]),min(abs(i+startx+0.5+TX-parcx[pp]),abs(i+startx+0.5-TX-parcx[pp])));
                        dy=min(abs(j+starty+0.5-parcy[pp]),min(abs(j+starty+0.5+TY-parcy[pp]),abs(j+starty+0.5-TY-parcy[pp])));
                        dz=min(abs(k+startz+0.5-parcz[pp]),min(abs(k+startz+0.5+TZ-parcz[pp]),abs(k+startz+0.5-TZ-parcz[pp])));

                        if(dx==abs(i+startx+0.5-parcx[pp])) dx=i+startx+0.5-parcx[pp];
                        if(dx==abs(i+startx+0.5+TX-parcx[pp])) dx=i+startx+0.5+TX-parcx[pp];
                        if(dx==abs(i+startx+0.5-TX-parcx[pp])) dx=i+startx+0.5-TX-parcx[pp];

                        if(dy==abs(j+starty+0.5-parcy[pp])) dy=j+starty+0.5-parcy[pp];
                        if(dy==abs(j+starty+0.5+TY-parcy[pp])) dy=j+starty+0.5+TY-parcy[pp];
                        if(dy==abs(j+starty+0.5-TY-parcy[pp])) dy=j+starty+0.5-TY-parcy[pp];

                        if(dz==abs(k+startz+0.5-parcz[pp])) dz=k+startz+0.5-parcz[pp];
                        if(dz==abs(k+startz+0.5+TZ-parcz[pp])) dz=k+startz+0.5+TZ-parcz[pp];
                        if(dz==abs(k+startz+0.5-TZ-parcz[pp])) dz=k+startz+0.5-TZ-parcz[pp];

                        Eulnwx[i][j][k]=dx;
                        Eulnwy[i][j][k]=dy;
                        Eulnwz[i][j][k]=dz;
                    }
                }
            }
}

void Infosendrecvparsolid()
{
    MPI_Status status;
    for(pp=0;pp<NP;pp++)
    {
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=parsolid[pp][NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=parsolid[pp][1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                parsolid[pp][NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                parsolid[pp][0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=parsolid[pp][i][NY-2][k];
                send_back[i+(k-1)*(NX)]=parsolid[pp][i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                parsolid[pp][i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                parsolid[pp][i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=parsolid[pp][i][j][NZ-2];
                send_down[i+j*(NX)]=parsolid[pp][i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                parsolid[pp][i][j][NZ-1]=recv_up[i+j*(NX)];
                parsolid[pp][i][j][0]=recv_down[i+j*(NX)];
            }
    }
    
}


void Infosendrecvuw()
{
    MPI_Status status;
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=uwx[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=uwx[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                uwx[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                uwx[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=uwx[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=uwx[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                uwx[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                uwx[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=uwx[i][j][NZ-2];
                send_down[i+j*(NX)]=uwx[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                uwx[i][j][NZ-1]=recv_up[i+j*(NX)];
                uwx[i][j][0]=recv_down[i+j*(NX)];
            }



        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=uwy[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=uwy[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                uwy[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                uwy[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=uwy[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=uwy[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                uwy[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                uwy[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=uwy[i][j][NZ-2];
                send_down[i+j*(NX)]=uwy[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                uwy[i][j][NZ-1]=recv_up[i+j*(NX)];
                uwy[i][j][0]=recv_down[i+j*(NX)];
            }



        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=uwz[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=uwz[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrS,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,taglS,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,taglS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                uwz[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                uwz[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=uwz[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=uwz[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfS,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbS,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbS,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                uwz[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                uwz[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=uwz[i][j][NZ-2];
                send_down[i+j*(NX)]=uwz[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,taguS,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,taguS,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagdS,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagdS,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                uwz[i][j][NZ-1]=recv_up[i+j*(NX)];
                uwz[i][j][0]=recv_down[i+j*(NX)];
            }
}

void Infosendrecvdvof()
{
    MPI_Status status;
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=dvofx[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=dvofx[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrdvx,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrdvx,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagldvx,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagldvx,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                dvofx[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                dvofx[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=dvofx[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=dvofx[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfdvx,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfdvx,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbdvx,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbdvx,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                dvofx[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                dvofx[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=dvofx[i][j][NZ-2];
                send_down[i+j*(NX)]=dvofx[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagudvx,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagudvx,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagddvx,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagddvx,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                dvofx[i][j][NZ-1]=recv_up[i+j*(NX)];
                dvofx[i][j][0]=recv_down[i+j*(NX)];
            }



        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=dvofy[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=dvofy[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrdvy,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrdvy,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagldvy,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagldvy,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                dvofy[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                dvofy[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=dvofy[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=dvofy[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfdvy,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfdvy,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbdvy,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbdvy,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                dvofy[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                dvofy[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=dvofy[i][j][NZ-2];
                send_down[i+j*(NX)]=dvofy[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagudvy,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagudvy,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagddvy,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagddvy,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                dvofy[i][j][NZ-1]=recv_up[i+j*(NX)];
                dvofy[i][j][0]=recv_down[i+j*(NX)];
            }



        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                send_right[j-1+(k-1)*(NY-2)]=dvofz[NX-2][j][k];
                send_left[j-1+(k-1)*(NY-2)]=dvofz[1][j][k];
            }
        MPI_Sendrecv(&send_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagrdvz,
                        &recv_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagrdvz,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_left[0],(NY-2)*(NZ-2),MPI_DOUBLE,destleft,tagldvz,
                        &recv_right[0],(NY-2)*(NZ-2),MPI_DOUBLE,destright,tagldvz,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(j=1;j<NY-1;j++)
            {
                dvofz[NX-1][j][k]=recv_right[j-1+(k-1)*(NY-2)];
                dvofz[0][j][k]=recv_left[j-1+(k-1)*(NY-2)];
            }

        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                send_front[i+(k-1)*(NX)]=dvofz[i][NY-2][k];
                send_back[i+(k-1)*(NX)]=dvofz[i][1][k];
            }
        MPI_Sendrecv(&send_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagfdvz,
                        &recv_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagfdvz,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_back[0],(NX)*(NZ-2),MPI_DOUBLE,destback,tagbdvz,
                        &recv_front[0],(NX)*(NZ-2),MPI_DOUBLE,destfront,tagbdvz,MPI_COMM_WORLD,&status);
        for(k=1;k<NZ-1;k++)
            for(i=0;i<NX;i++)
            {
                dvofz[i][NY-1][k]=recv_front[i+(k-1)*(NX)];
                dvofz[i][0][k]=recv_back[i+(k-1)*(NX)];
            }

        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                send_up[i+j*(NX)]=dvofz[i][j][NZ-2];
                send_down[i+j*(NX)]=dvofz[i][j][1];
            }
        MPI_Sendrecv(&send_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagudvz,
                        &recv_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagudvz,MPI_COMM_WORLD,&status);
        MPI_Sendrecv(&send_down[0],(NX)*(NY),MPI_DOUBLE,destdown,tagddvz,
                        &recv_up[0],(NX)*(NY),MPI_DOUBLE,destup,tagddvz,MPI_COMM_WORLD,&status);
        for(j=0;j<NY;j++)
            for(i=0;i<NX;i++)
            {
                dvofz[i][j][NZ-1]=recv_up[i+j*(NX)];
                dvofz[i][j][0]=recv_down[i+j*(NX)];
            }
}


void Parsolidupdate()
{
    for(i=0;i<NX;i++)
        for(j=0;j<NY;j++)
            for(k=0;k<NZ;k++)
            {
                solid0[i][j][k]=solid[i][j][k];
                uwx0[i][j][k]=uwx[i][j][k];
                uwy0[i][j][k]=uwy[i][j][k];
                uwz0[i][j][k]=uwz[i][j][k];
                uwx[i][j][k]=0.0;
                uwy[i][j][k]=0.0;
                uwz[i][j][k]=0.0;
                if(solid[i][j][k]<2.0)
                {
                    solid[i][j][k]=0.0;
                    for(pp=0;pp<NP;pp++)
                    {
                        if(parsolid[pp][i][j][k]==2.0)
                        {
                            solid[i][j][k]=1.0;
                            uwx[i][j][k]=paruwx[pp][i][j][k];
                            uwy[i][j][k]=paruwy[pp][i][j][k];
                            uwz[i][j][k]=paruwz[pp][i][j][k];
                        }
                    }
                }
                
            }
    int irf, jrf, krf;
    double wsum;
    double mloss, madd;
    double nloss, nadd;
    double naddsum;

    
    for(pp=0;pp<NP;pp++)
    {
        mloss=0.0;
        madd=0.0;
        nloss=0.0;
        nadd=0.0;
        naddsum=0.0;
        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(solid0[i][j][k]==0.0&&parsolid[pp][i][j][k]==2.0)
                    {
                        mloss=mloss+vof[i][j][k];
                    }
                }
        MPI_Allreduce(&mloss, &mlosstotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(parsolid0[pp][i][j][k]==2.0&&solid[i][j][k]==0.0)
                    {
                        ux[i][j][k]=uwx0[i][j][k]*0.0;
                        uy[i][j][k]=uwy0[i][j][k]*0.0;
                        uz[i][j][k]=uwz0[i][j][k]*0.0;
                        vof[i][j][k]=0.0;
                        p[i][j][k]=0.0;
                        wsum=0.0;
                        for(l=1;l<Q;l++)
                        {
                            irf=i+ex[l];
                            jrf=j+ey[l];
                            krf=k+ez[l];

                            if(solid0[irf][jrf][krf]==0.0)
                            {
                                vof[i][j][k]=vof[i][j][k]+w[l]*vof[irf][jrf][krf];
                                p[i][j][k]=p[i][j][k]+w[l]*p[irf][jrf][krf];
                                wsum=wsum+w[l];
                            }
                        }
                        if(wsum>0.0)
                        {
                            p[i][j][k]=p[i][j][k]/wsum;
                            vof[i][j][k]=vof[i][j][k]/wsum;
                        }
                        else
                        {
                            p[i][j][k]=0.0;
                            vof[i][j][k]=0.0;
                        }
                        vof[i][j][k]=max(0.0,min(1.0,vof[i][j][k]));
                        rho[i][j][k]=rhog+(vof[i][j][k]-vofg)/(vofl-vofg)*(rhol-rhog);
                        madd=madd+vof[i][j][k];

                        for(l=0;l<Q;l++)
                        {
                            fA[i][j][k][l]=feq(l,rho[i][j][k],p[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                            fB[i][j][k][l]=fbeq(l,vof[i][j][k],ux[i][j][k],uy[i][j][k],uz[i][j][k]);
                        }
                    }
                }
        MPI_Allreduce(&madd, &maddtotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    msourcetemp[i][j][k]=0.0;
                    if(solid[i][j][k]==0.0&&vof[i][j][k]>0.9)
                    {
                        for(l=1;l<Q;l++)
                        {
                            irf=i+ex[l];
                            jrf=j+ey[l];
                            krf=k+ez[l];
                            if(parsolid[pp][irf][jrf][krf]==2.0)
                            {
                                naddsum=naddsum+1.0;
                                msourcetemp[i][j][k]=mlosstotal-maddtotal;
                                break;
                            }
                        }
                    }
                }
        MPI_Allreduce(&naddsum, &ntotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ntotal=max(1.0,ntotal);

        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    msource[i][j][k]=msource[i][j][k]+msourcetemp[i][j][k]/ntotal;
                }
    }
    
    Infosendrecvmacro();
    Infosendrecvsolid();
    Getbounce();
    Phisolid();
    Infosendrecvmacro();   
}

void Movement()
{
    double lim=10.0;
    for(pp=0;pp<NP;pp++)
    {
        parFtotalx[pp]=parFcx[pp]+parFhx[pp]+Fpqx[pp]+mpar[pp]*gvx;
        parFtotaly[pp]=parFcy[pp]+parFhy[pp]+Fpqy[pp]+mpar[pp]*gvy;
        parFtotalz[pp]=parFcz[pp]+parFhz[pp]+Fpqz[pp]+mpar[pp]*gvz;
        parTtotalx[pp]=parTcx[pp]+parThx[pp]+Tpqx[pp];
        parTtotaly[pp]=parTcy[pp]+parThy[pp]+Tpqy[pp];
        parTtotalz[pp]=parTcz[pp]+parThz[pp]+Tpqz[pp];

        parcx[pp]=parcx[pp]+dtD*parux[pp]+dtD*dtD*parFtotalx[pp]/mpar[pp]/2.0;
        parcy[pp]=parcy[pp]+dtD*paruy[pp]+dtD*dtD*parFtotaly[pp]/mpar[pp]/2.0;
        parcz[pp]=parcz[pp]+dtD*paruz[pp]+dtD*dtD*parFtotalz[pp]/mpar[pp]/2.0;

        parux[pp]=parux[pp]+dtD*parFtotalx[pp]/mpar[pp];
        paruy[pp]=paruy[pp]+dtD*parFtotaly[pp]/mpar[pp];
        paruz[pp]=paruz[pp]+dtD*parFtotalz[pp]/mpar[pp];
        parwx[pp]=parwx[pp]+dtD*parTtotalx[pp]/Ipar[pp];
        parwy[pp]=parwy[pp]+dtD*parTtotaly[pp]/Ipar[pp];
        parwz[pp]=parwz[pp]+dtD*parTtotalz[pp]/Ipar[pp];

        parcx[pp]=parcx[pp]-TX*floor(parcx[pp]/TX);
        parcy[pp]=parcy[pp]-TY*floor(parcy[pp]/TY);
        parcz[pp]=parcz[pp]-TZ*floor(parcz[pp]/TZ);
        
    }
}

void Hydroforce()
{
    int ih, jh, kh;
    double dfx, dfy, dfz;
    double temp1,temp2;
    double dx, dy, dz;
    for(pp=0;pp<NP;pp++)
    {
        parFhx[pp]=0.0;
        parFhy[pp]=0.0;
        parFhz[pp]=0.0;
        parThx[pp]=0.0;
        parThy[pp]=0.0;
        parThz[pp]=0.0;

        for(i=1;i<NX-1;i++)
            for(j=1;j<NY-1;j++)
                for(k=1;k<NZ-1;k++)
                {
                    if(parsolid[pp][i][j][k]==2.0)
                    {
                        for(l=1;l<Q;l++)
                        {
                            ih=i+ex[l];
                            jh=j+ey[l];
                            kh=k+ez[l];
                            if(solid[ih][jh][kh]==0.0)
                            {
                                dfx=(ex[op[l]]-uwx[i][j][k])*fApost[ih][jh][kh][op[l]]-(ex[l]-uwx[i][j][k])*fApost[i][j][k][l];
                                dfy=(ey[op[l]]-uwy[i][j][k])*fApost[ih][jh][kh][op[l]]-(ey[l]-uwy[i][j][k])*fApost[i][j][k][l];
                                dfz=(ez[op[l]]-uwz[i][j][k])*fApost[ih][jh][kh][op[l]]-(ez[l]-uwz[i][j][k])*fApost[i][j][k][l];

                                dfx=dfx*3.0;
                                dfy=dfy*3.0;
                                dfz=dfz*3.0;
                                parFhx[pp]=parFhx[pp]+dfx;
                                parFhy[pp]=parFhy[pp]+dfy;
                                parFhz[pp]=parFhz[pp]+dfz;

                                dx=min(abs(i+startx+0.5-parcx[pp]),min(abs(i+startx+0.5+TX-parcx[pp]),abs(i+startx+0.5-TX-parcx[pp])));
                                dy=min(abs(j+starty+0.5-parcy[pp]),min(abs(j+starty+0.5+TY-parcy[pp]),abs(j+starty+0.5-TY-parcy[pp])));
                                dz=min(abs(k+startz+0.5-parcz[pp]),min(abs(k+startz+0.5+TZ-parcz[pp]),abs(k+startz+0.5-TZ-parcz[pp])));

                                if(dx==abs(i+startx+0.5-parcx[pp])) dx=i+startx+0.5-parcx[pp];
                                if(dx==abs(i+startx+0.5+TX-parcx[pp])) dx=i+startx+0.5+TX-parcx[pp];
                                if(dx==abs(i+startx+0.5-TX-parcx[pp])) dx=i+startx+0.5-TX-parcx[pp];

                                if(dy==abs(j+starty+0.5-parcy[pp])) dy=j+starty+0.5-parcy[pp];
                                if(dy==abs(j+starty+0.5+TY-parcy[pp])) dy=j+starty+0.5+TY-parcy[pp];
                                if(dy==abs(j+starty+0.5-TY-parcy[pp])) dy=j+starty+0.5-TY-parcy[pp];

                                if(dz==abs(k+startz+0.5-parcz[pp])) dz=k+startz+0.5-parcz[pp];
                                if(dz==abs(k+startz+0.5+TZ-parcz[pp])) dz=k+startz+0.5+TZ-parcz[pp];
                                if(dz==abs(k+startz+0.5-TZ-parcz[pp])) dz=k+startz+0.5-TZ-parcz[pp];

                                parThx[pp]=parThx[pp]+VCross(dx,dy,dz,dfx,dfy,dfz,0);
                                parThy[pp]=parThy[pp]+VCross(dx,dy,dz,dfx,dfy,dfz,1);
                                parThz[pp]=parThz[pp]+VCross(dx,dy,dz,dfx,dfy,dfz,2);
                            }
                        }
                    }
                }
                temp1=parFhx[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parFhx[pp]=temp2;

                temp1=parFhy[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parFhy[pp]=temp2;

                temp1=parFhz[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parFhz[pp]=temp2;

                temp1=parThx[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parThx[pp]=temp2;

                temp1=parThy[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parThy[pp]=temp2;

                temp1=parThz[pp];
                MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                parThz[pp]=temp2;
    }
}

void Capillaryforce()
{
    double dvx, dvy, dvz;
    double dwx, dwy, dwz;
    double vlen;
    double vwx, vwy, vwz;
    double mx, my, mz;
    double dfx, dfy, dfz;
    double temp1, temp2;

    int iv,jv,kv;
    double wsumv;
    double Lagvof;
    Infosendrecvdvof();
    for(pp=0;pp<NP;pp++)
    {
        parFcx[pp]=0.0;
        parFcy[pp]=0.0;
        parFcz[pp]=0.0;
        parTcx[pp]=0.0;
        parTcy[pp]=0.0;
        parTcz[pp]=0.0;

        for(pl=0;pl<NLag;pl++)
        {
            if(Lagx[pp][pl]>=(1.0+startx)&&Lagx[pp][pl]<(NX-1.0+startx))
            {
                if(Lagy[pp][pl]>=(1.0+starty)&&Lagy[pp][pl]<(NY-1.0+starty))
                {
                    if(Lagz[pp][pl]>=(1.0+startz)&&Lagz[pp][pl]<(NZ-1.0+startz))
                    {
                        dvx=0.0;
                        dvy=0.0;
                        dvz=0.0;
                        Lagvof=0.0;

                        if(solid[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]]==0.0)
                        {
                            dvx=dvofx[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                            dvy=dvofy[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                            dvz=dvofz[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                            Lagvof=vof[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                        }

                        if(solid[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]]>0.0)
                        {
                            wsumv=0.0;
                            for(l=1;l<Q;l++)
                            {
                                iv=Eulx[pp][pl]+ex[l];
                                jv=Euly[pp][pl]+ey[l];
                                kv=Eulz[pp][pl]+ez[l];
                                if(solid[iv][jv][kv]==0.0)
                                {
                                    dvx=dvx+w[l]*dvofx[iv][jv][kv];
                                    dvy=dvy+w[l]*dvofy[iv][jv][kv];
                                    dvz=dvz+w[l]*dvofz[iv][jv][kv];
                                    Lagvof=Lagvof+w[l]*vof[iv][jv][kv];
                                    wsumv=wsumv+w[l];
                                }
                            }

                            if(wsumv>0.0)
                            {
                                dvx=dvx/wsumv;
                                dvy=dvy/wsumv;
                                dvz=dvz/wsumv;
                                Lagvof=Lagvof/wsumv;
                            }
                            else
                            {
                                dvx=0.0;
                                dvy=0.0;
                                dvz=0.0;
                                Lagvof=0.0;
                            }
                        }
                        
                        dwx=parnwx[pp][pl];
                        dwy=parnwy[pp][pl];
                        dwz=parnwz[pp][pl];
                        vlen=dvx*dvx+dvy*dvy+dvz*dvz;
                        if(vlen<1.0e-12)
                        {
                            dvx=0.0;
                            dvy=0.0;
                            dvz=0.0;
                        }
                        else
                        {
                            dvx=dvx/sqrt(vlen);
                            dvy=dvy/sqrt(vlen);
                            dvz=dvz/sqrt(vlen);
                        }

                        vwx=VCross(dvx,dvy,dvz,dwx,dwy,dwz,0);
                        vwy=VCross(dvx,dvy,dvz,dwx,dwy,dwz,1);
                        vwz=VCross(dvx,dvy,dvz,dwx,dwy,dwz,2);

                        mx=VCross(vwx,vwy,vwz,dvx,dvy,dvz,0);
                        my=VCross(vwx,vwy,vwz,dvx,dvy,dvz,1);
                        mz=VCross(vwx,vwy,vwz,dvx,dvy,dvz,2);

                        dfx=cos(0.5*acos(-1)-theta)*mx*Lagds[pp][pl]*3.0*sigma/2.0/vofw*pow((4.0*Lagvof*Lagvof-4.0*Lagvof),2.0);
                        dfy=cos(0.5*acos(-1)-theta)*my*Lagds[pp][pl]*3.0*sigma/2.0/vofw*pow((4.0*Lagvof*Lagvof-4.0*Lagvof),2.0);
                        dfz=cos(0.5*acos(-1)-theta)*mz*Lagds[pp][pl]*3.0*sigma/2.0/vofw*pow((4.0*Lagvof*Lagvof-4.0*Lagvof),2.0);
                        
                        /*dfx=dfx-dwx*Lagds[pp][pl]*p[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                        dfy=dfy-dwy*Lagds[pp][pl]*p[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];
                        dfz=dfz-dwz*Lagds[pp][pl]*p[Eulx[pp][pl]][Euly[pp][pl]][Eulz[pp][pl]];*/

                        parFcx[pp]=parFcx[pp]+dfx;
                        parFcy[pp]=parFcy[pp]+dfy;
                        parFcz[pp]=parFcz[pp]+dfz;

                        parTcx[pp]=parTcx[pp]+VCross(parr[pp]*dwx,parr[pp]*dwy,parr[pp]*dwz,dfx,dfy,dfz,0);
                        parTcy[pp]=parTcy[pp]+VCross(parr[pp]*dwx,parr[pp]*dwy,parr[pp]*dwz,dfx,dfy,dfz,1);
                        parTcz[pp]=parTcz[pp]+VCross(parr[pp]*dwx,parr[pp]*dwy,parr[pp]*dwz,dfx,dfy,dfz,2);

                    }
                }
            }
        }
        temp1=parFcx[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parFcx[pp]=temp2;

        temp1=parFcy[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parFcy[pp]=temp2;

        temp1=parFcz[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parFcz[pp]=temp2;

        temp1=parTcx[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parTcx[pp]=temp2;

        temp1=parTcy[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parTcy[pp]=temp2;

        temp1=parTcz[pp];
        MPI_Allreduce(&temp1,&temp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        parTcz[pp]=temp2;
    }
}

void fOutputVTK3D()
{ 
    // output .vts file with properties for all fluids 
    int istart, iend, jstart, jend, kstart, kend;
    int offset;
    float ipos[3];
    long kpos;
    int ilen=0, ilen1=0, ilen2=0, ilen3=0;
    char name[80];
 
    ostringstream command;
    command<<"mkdir -p "<<dataSp<<dataSp1<<n;
    system(command.str().c_str());

    sprintf(name, "%s%s%d%s%d", dataSp, dataSp1, n, dataSp2, mpirank);
    ofstream file(name);
 
    file<<"<?xml version=\"1.0\"?>"<<endl; 
    file<<"<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"BigEndian\">"<<endl;

    // determine extent of piece 
    istart = 0; 
    iend = NX-1; 
    if(rankx == 0) istart = 1;
 
    jstart = 0; 
    jend = NY-1; 
    if(ranky == 0) jstart = 1;
 
    kstart = 0; 
    kend = NZ-1; 
    if(rankz == 0) kstart = 1;
 
    //file<<"<StructuredGrid WholeExtent=\"";
    file<<"<RectilinearGrid WholeExtent=\"";
    file<<(startx+istart)<<" "<<(startx+iend-1)<<" "
        <<(starty+jstart)<<" "<<(starty+jend-1)<<" "
        <<(startz+kstart)<<" "<<(startz+kend-1)<<"\">"<<endl;
    file<<"<Piece Extent=\""<<(startx+istart)<<" "<<(startx+iend-1)<<" "
                            <<(starty+jstart)<<" "<<(starty+jend-1)<<" "
                            <<(startz+kstart)<<" "<<(startz+kend-1)<<"\">"<<endl;
    file<<"<PointData Scalars=\"phase_field\" Vectors=\"velocity\">"<<endl;

    // determine number of data points 
    ilen1=NX-1;
    if(rankx == 0) ilen1 -= 1; 

    ilen2=NY-1;
    if(ranky == 0) ilen2 -= 1; 

    ilen3=NZ-1;
    if(rankz == 0) ilen3 -= 1; 

    ////////////////////////
    offset = 0;

    file<<"<DataArray Name=\"VOF\" type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    offset += sizeof(float)*ilen1*ilen2*ilen3 + sizeof(unsigned int);
    file<<"</DataArray>"<<endl;
    
    file<<"<DataArray Name=\"Solid1\" type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    offset += sizeof(float)*ilen1*ilen2*ilen3 + sizeof(unsigned int);
    file<<"</DataArray>"<<endl;

    file<<"<DataArray Name=\"Solid2\" type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    offset += sizeof(float)*ilen1*ilen2*ilen3 + sizeof(unsigned int);
    file<<"</DataArray>"<<endl;

    file<<"<DataArray Name=\"T\" type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    offset += sizeof(float)*ilen1*ilen2*ilen3 + sizeof(unsigned int);
    file<<"</DataArray>"<<endl;

    file<<"<DataArray Name=\"u\" type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    offset += sizeof(float) * 3 * ilen1 * ilen2 * ilen3 + sizeof(unsigned int);
    file<<"</DataArray>"<<endl;

 
    file<<"</PointData>"<<endl;
    //  file<<"<Points>"<<endl;
    file<<"<Coordinates>"<<endl;

    // file<<"<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    // offset+=sizeof(float)*ilen1+sizeof(unsigned int);
    file<<"<DataArray type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    file<<"</DataArray>"<<endl;

    offset+=sizeof(float)*ilen1+sizeof(unsigned int);
    file<<"<DataArray type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    file<<"</DataArray>"<<endl;

    offset+=sizeof(float)*ilen2+sizeof(unsigned int);
    file<<"<DataArray type=\"Float32\" format=\"appended\" offset=\""<<offset<<"\">"<<endl;
    file<<"</DataArray>"<<endl;
    
    //file<<"</Points>"<<endl;
    file<<"</Coordinates>"<<endl;

    file<<"</Piece>"<<endl;
    //  file<<"</StructuredGrid>"<<endl;
    file<<"</RectilinearGrid>"<<endl;

    file<<"<AppendedData encoding=\"raw\">"<<endl;
    file<<"_";  

    /////////////////////////
    // write vof
    kpos=sizeof(float)*ilen1*ilen2*ilen3;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(k=kstart; k<kend; k=k+1)  
    for(j=jstart; j<jend; j=j+1)  
    for(i=istart; i<iend; i=i+1)
    { 
        ipos[0]=float(vof[i][j][k]); 
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float)); 
    }
 
    // write solid1
    kpos=sizeof(float)*ilen1*ilen2*ilen3;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(k=kstart; k<kend; k=k+1)  
    for(j=jstart; j<jend; j=j+1)  
    for(i=istart; i<iend; i=i+1)
    { 
        if(solid[i][j][k]==2.0) ipos[0]=float(solid[i][j][k]); 
        if(solid[i][j][k]<=1.0) ipos[0]=0.0;
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float)); 
    }

    // write solid2
    kpos=sizeof(float)*ilen1*ilen2*ilen3;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(k=kstart; k<kend; k=k+1)  
    for(j=jstart; j<jend; j=j+1)  
    for(i=istart; i<iend; i=i+1)
    { 
        if(solid[i][j][k]==2.0) ipos[0]=0.0; 
        if(solid[i][j][k]<=1.0) ipos[0]=float(solid[i][j][k]); 
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float)); 
    }

    // write T
    kpos=sizeof(float)*ilen1*ilen2*ilen3;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(k=kstart; k<kend; k=k+1)  
    for(j=jstart; j<jend; j=j+1)  
    for(i=istart; i<iend; i=i+1)
    { 
        ipos[0]=float(T[i][j][k]); 
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float)); 
    }

    // write u vector (ux, uy, uz)
    kpos = sizeof(float) * 3 * ilen1 * ilen2 * ilen3;
    if (!bigend) fByteSwap(&kpos, sizeof(unsigned int), 1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for (k = kstart; k < kend; k++) 
    for (j = jstart; j < jend; j++) 
    for (i = istart; i < iend; i++) 
    {
        ipos[0] = float(ux[i][j][k]); // ux
        ipos[1] = float(uy[i][j][k]); // uy
        ipos[2] = float(uz[i][j][k]); // uz

        if (!bigend) fByteSwap(ipos, sizeof(float), 3);
        file.write(reinterpret_cast<const char*>(ipos), sizeof(float) * 3);
    }
 
    /////////////////////////
    kpos=sizeof(float)*ilen1;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(i=istart; i<iend; i=i+1)
    { 
        ipos[0]=(startx + i)*float(dx);
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float));
    }

    kpos=sizeof(float)*ilen2;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(j=jstart; j<jend; j=j+1)
    { 
        ipos[0]=(starty + j)*float(dx);
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float));
    }

    kpos=sizeof(float)*ilen3;
    if (!bigend) fByteSwap(&kpos,sizeof(unsigned int),1);
    file.write(reinterpret_cast<const char*>(&kpos), sizeof(unsigned int));
    for(k=kstart; k<kend; k=k+1)
    { 
        ipos[0]=(startz + k)*float(dx);
        if (!bigend) fByteSwap(ipos,sizeof(float),1);
        file.write((char*)&ipos[0], sizeof(float));
    }

    ////////////////////
    file<<"</AppendedData>"<<endl;
    file<<"</VTKFile>"<<endl;
    
    file.close(); 
}

void fOutputInfo()
{
    int istart, iend, jstart, jend, kstart, kend;
    int buf[7], bufc[7], bufAll[PXYZ][7]; 
    int np, npn;

    MPI_Status status;

    // determine extent of piece 
    istart = 0; 
    iend = NX-1;
    if(rankx == 0) istart = 1; 

    jstart = 0; 
    jend = NY-1; 
    if(ranky == 0) jstart = 1; 

    kstart = 0; 
    kend = NZ-1; 
    if(rankz == 0) kstart = 1; 
    
    buf[0]=mpirank;
    buf[1]=startx+istart;
    buf[2]=startx+iend-1;
    buf[3]=starty+jstart;
    buf[4]=starty+jend-1;
    buf[5]=startz+kstart;
    buf[6]=startz+kend-1; 
 
    // output system information  
    if(mpirank != 0)
    {
        // send information to process 0   
        MPI_Send(&buf[0], 7, MPI_INT, 0, 14159, MPI_COMM_WORLD);  
    }
    if(mpirank == 0)
    {
        for(npn=0; npn<7; npn++) bufAll[0][npn] = buf[npn];  

        // receive information from other processes and print
        for(np=1; np<PXYZ; np++)
        {
            MPI_Recv(&bufc[0], 7, MPI_INT, np, 14159, MPI_COMM_WORLD, &status);

            for(npn=0; npn<7; npn++) bufAll[np][npn] = bufc[npn]; 
        }
    }
    
    ostringstream command;
    command<<"mkdir -p "<<dataF;
    system(command.str().c_str());
 
    if(mpirank == 0)
    { 
        FILE *fp_out; 
        char fnQ1[88];

        ostringstream command;
        command<<"mkdir -p "<<dataF;
        system(command.str().c_str());
 
        ///////////////////info/////////////////
        sprintf(fnQ1,"%s%s", dataF, dataF2);
        if(( fp_out=fopen(fnQ1,"w"))==NULL) { printf(" File Open Error 2\n");exit(1);} 
        fprintf(fp_out,"numberofDimensions    %d \n", 3);
        fprintf(fp_out,"sizeofSystem          %d \n", PXYZ); 
        fclose(fp_out);
 
        ///////////////////ext/////////////////
        sprintf(fnQ1,"%s%s", dataF, dataF3);
        if( ( fp_out=fopen(fnQ1,"w"))==NULL) { printf(" File Open Error 3\n");exit(1); }
        for(np=0; np<PXYZ; np++) 
            fprintf(fp_out,"extent_%d %d %d %d %d %d %d \n", np, bufAll[np][1], bufAll[np][2], bufAll[np][3], bufAll[np][4], bufAll[np][5], bufAll[np][6]);
        fclose(fp_out);
    } 
}

int fBigEndian()
{
    // indicate endianness of machine: 1 = big endian, 0 = little endian
    short int n = 1;
    char *ep = (char *)&n;

    return (*ep == 0);
}

void fByteSwap(void *data, int len, int count)
{
    // swap byte order to select endian type for binary files

    char tmp, *cdat = (char *) data;
    int k;
 
    if (len==1) return;
    else if (len==2) while (count--) 
    {
        tmp = cdat[0];  cdat[0] = cdat[1];  cdat[1] = tmp;
        cdat += 2;
    }
    else if (len==4) while (count--)
    {
        tmp = cdat[0];  cdat[0] = cdat[3];  cdat[3] = tmp;
        tmp = cdat[1];  cdat[1] = cdat[2];  cdat[2] = tmp;
        cdat += 4;
    }
    else if (len==8) while (count--)
    {
        tmp = cdat[0];  cdat[0] = cdat[7];  cdat[7] = tmp;
        tmp = cdat[1];  cdat[1] = cdat[6];  cdat[6] = tmp;
        tmp = cdat[2];  cdat[2] = cdat[5];  cdat[5] = tmp;
        tmp = cdat[3];  cdat[3] = cdat[4];  cdat[4] = tmp;
        cdat += 8;
    }
    else 
    {
        for(k=0; k<len/2; k++) 
        {
            tmp = cdat[k];
            cdat[k] = cdat[len-1-k];
            cdat[len-1-k] = tmp;
        }
    }
}

void Contactforce()
{
    for (pp=0;pp<NP;pp++)
    {
        Fpqx[pp]=0.0;
        Fpqy[pp]=0.0;
        Fpqz[pp]=0.0;

        Tpqx[pp]=0.0;
        Tpqy[pp]=0.0;
        Tpqz[pp]=0.0;

        for (qq=0;qq<NP;qq++)
        {
            if(pp!=qq)
            {
                dpqx=min(abs(parcx[pp]-parcx[qq]),min(abs(parcx[pp]+TX-parcx[qq]),abs(parcx[pp]-TX-parcx[qq])));
                dpqy=min(abs(parcy[pp]-parcy[qq]),min(abs(parcy[pp]+TY-parcy[qq]),abs(parcy[pp]-TY-parcy[qq])));
                dpqz=min(abs(parcz[pp]-parcz[qq]),min(abs(parcz[pp]+TZ-parcz[qq]),abs(parcz[pp]-TZ-parcz[qq])));

                if(dpqx==abs(parcx[pp]-parcx[qq])) dpqx=parcx[pp]-parcx[qq];
                if(dpqx==abs(parcx[pp]+TX-parcx[qq])) dpqx=parcx[pp]+TX-parcx[qq];
                if(dpqx==abs(parcx[pp]-TX-parcx[qq])) dpqx=parcx[pp]-TX-parcx[qq];

                if(dpqy==abs(parcy[pp]-parcy[qq])) dpqy=parcy[pp]-parcy[qq];
                if(dpqy==abs(parcy[pp]+TY-parcy[qq])) dpqy=parcy[pp]+TY-parcy[qq];
                if(dpqy==abs(parcy[pp]-TY-parcy[qq])) dpqy=parcy[pp]-TY-parcy[qq];

                if(dpqz==abs(parcz[pp]-parcz[qq])) dpqz=parcz[pp]-parcz[qq];
                if(dpqz==abs(parcz[pp]+TZ-parcz[qq])) dpqz=parcz[pp]+TZ-parcz[qq];
                if(dpqz==abs(parcz[pp]-TZ-parcz[qq])) dpqz=parcz[pp]-TZ-parcz[qq];
                

                deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp]-parr[qq];
                if(deltapq<0.0)
                {
                    Fn=0.0;
                    Ftx=0.0;
                    Fty=0.0;
                    Ftz=0.0;

                    npqx=dpqx;
                    npqy=dpqy;
                    npqz=dpqz;

                    npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
                    if(npqlen>0.0)
                    {
                        npqx=npqx/npqlen;
                        npqy=npqy/npqlen;
                        npqz=npqz/npqlen;
                    }
                    else
                    {
                        npqx=0.0;
                        npqy=0.0;
                        npqz=0.0;
                    }
                    

                    /*cpqx=parcx[qq]+(parr[qq]+0.0*deltapq)*npqx;
                    cpqy=parcy[qq]+(parr[qq]+0.0*deltapq)*npqy;
                    cpqz=parcz[qq]+(parr[qq]+0.0*deltapq)*npqz;

                    while(cpqx<0.0) cpqx=cpqx+TX;
                    while(cpqx>=TX) cpqx=cpqx-TX;
                    while(cpqy<0.0) cpqy=cpqy+TY;
                    while(cpqy>=TY) cpqy=cpqy-TY;
                    while(cpqz<0.0) cpqz=cpqz+TZ;
                    while(cpqz>=TZ) cpqz=cpqz-TZ;*/

                    upqx=parux[pp]-parux[qq]-VCross((parr[pp]+0.0*deltapq)*parwx[pp]+(parr[qq]+0.0*deltapq)*parwx[qq],(parr[pp]+0.0*deltapq)*parwy[pp]+(parr[qq]+0.0*deltapq)*parwy[qq],(parr[pp]+0.0*deltapq)*parwz[pp]+(parr[qq]+0.0*deltapq)*parwz[qq],npqx,npqy,npqz,0);
                    upqy=paruy[pp]-paruy[qq]-VCross((parr[pp]+0.0*deltapq)*parwx[pp]+(parr[qq]+0.0*deltapq)*parwx[qq],(parr[pp]+0.0*deltapq)*parwy[pp]+(parr[qq]+0.0*deltapq)*parwy[qq],(parr[pp]+0.0*deltapq)*parwz[pp]+(parr[qq]+0.0*deltapq)*parwz[qq],npqx,npqy,npqz,1);
                    upqz=paruz[pp]-paruz[qq]-VCross((parr[pp]+0.0*deltapq)*parwx[pp]+(parr[qq]+0.0*deltapq)*parwx[qq],(parr[pp]+0.0*deltapq)*parwy[pp]+(parr[qq]+0.0*deltapq)*parwy[qq],(parr[pp]+0.0*deltapq)*parwz[pp]+(parr[qq]+0.0*deltapq)*parwz[qq],npqx,npqy,npqz,2);

                    upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

                    upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
                    upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
                    upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

                    deltapqtx[pp][qq]=deltapqtx0[pp][qq]-(deltapqtx0[pp][qq]*npqx+deltapqty0[pp][qq]*npqy+deltapqtz0[pp][qq]*npqz)*npqx;
                    deltapqty[pp][qq]=deltapqty0[pp][qq]-(deltapqtx0[pp][qq]*npqx+deltapqty0[pp][qq]*npqy+deltapqtz0[pp][qq]*npqz)*npqy;
                    deltapqtz[pp][qq]=deltapqtz0[pp][qq]-(deltapqtx0[pp][qq]*npqx+deltapqty0[pp][qq]*npqy+deltapqtz0[pp][qq]*npqz)*npqz;

                    deltapqtlen=sqrt(deltapqtx[pp][qq]*deltapqtx[pp][qq]+deltapqty[pp][qq]*deltapqty[pp][qq]+deltapqtz[pp][qq]*deltapqtz[pp][qq]);
                    deltapqtlen0=sqrt(deltapqtx0[pp][qq]*deltapqtx0[pp][qq]+deltapqty0[pp][qq]*deltapqty0[pp][qq]+deltapqtz0[pp][qq]*deltapqtz0[pp][qq]);

                    if(deltapqtlen>1.0e-12)
                    {
                        deltapqtx[pp][qq]=deltapqtx[pp][qq]*deltapqtlen0/deltapqtlen;
                        deltapqty[pp][qq]=deltapqty[pp][qq]*deltapqtlen0/deltapqtlen;
                        deltapqtz[pp][qq]=deltapqtz[pp][qq]*deltapqtlen0/deltapqtlen;
                    }

                    /*deltapqtx[pp][qq]=deltapqtx0[pp][qq];
                    deltapqty[pp][qq]=deltapqty0[pp][qq];
                    deltapqtz[pp][qq]=deltapqtz0[pp][qq];*/

                    Fn=-kn*deltapq-gamman*((parux[pp]-parux[qq])*npqx+(paruy[pp]-paruy[qq])*npqy+(paruz[pp]-paruz[qq])*npqz);
                    Fttestx=-kt*deltapqtx[pp][qq]-gammat*upqtx;
                    Fttesty=-kt*deltapqty[pp][qq]-gammat*upqty;
                    Fttestz=-kt*deltapqtz[pp][qq]-gammat*upqtz;

                    if(Fn<0.0) Fn=0.0;

                    Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
                    Fttestlen=max(1.0e-12,Fttestlen);

                    if(Fttestlen<=muf*Fn)
                    {
                        deltapqtx0[pp][qq]=deltapqtx[pp][qq]+upqtx*dtD;
                        deltapqty0[pp][qq]=deltapqty[pp][qq]+upqty*dtD;
                        deltapqtz0[pp][qq]=deltapqtz[pp][qq]+upqtz*dtD;

                        Ftx=Fttestx;
                        Fty=Fttesty;
                        Ftz=Fttestz;
                    }
                    else
                    {
                        deltapqtx0[pp][qq]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                        deltapqty0[pp][qq]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                        deltapqtz0[pp][qq]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                        Ftx=muf*Fn*Fttestx/Fttestlen;
                        Fty=muf*Fn*Fttesty/Fttestlen;
                        Ftz=muf*Fn*Fttestz/Fttestlen;
                    }

                Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
                Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
                Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

                Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.0*deltapq)*npqx,-(parr[pp]+0.0*deltapq)*npqy,-(parr[pp]+0.0*deltapq)*npqz, Ftx, Fty, Ftz, 0);
                Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.0*deltapq)*npqx,-(parr[pp]+0.0*deltapq)*npqy,-(parr[pp]+0.0*deltapq)*npqz, Ftx, Fty, Ftz, 1);
                Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.0*deltapq)*npqx,-(parr[pp]+0.0*deltapq)*npqy,-(parr[pp]+0.0*deltapq)*npqz, Ftx, Fty, Ftz, 2);

                }
                else
                {
                    deltapqtx0[pp][qq]=0.0;
                    deltapqty0[pp][qq]=0.0;
                    deltapqtz0[pp][qq]=0.0;
                }
                if(n%1000==0&&pp==0&&qq==1&&mpirank==0)
                {
                    cout<<n<<" "<<deltapq<<" "<<Fn<<" "<<Ftx<<" "<<Fty<<" "<<Ftz<<" "<<deltapqtx0[pp][qq]<<" "<<deltapqty0[pp][qq]<<" "<<deltapqtz0[pp][qq]<<endl;
                }

            }
        }
        Parboundaryfront();
        Parboundaryback();
        Parboundarytop();
        Parboundarybottom();
        Parboundaryleft();
        Parboundaryright();
    }
}

void Parboundaryfront()
{
    if(parcy[pp]-1.0<=parr[pp])
        {
            dpqx=0.0;
            dpqy=parcy[pp]-1.0;
            dpqz=0.0;
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqtfrontx[pp]=deltapqtfrontx0[pp]-(deltapqtfrontx0[pp]*npqx+deltapqtfronty0[pp]*npqy+deltapqtfrontz0[pp]*npqz)*npqx;
            deltapqtfronty[pp]=deltapqtfronty0[pp]-(deltapqtfrontx0[pp]*npqx+deltapqtfronty0[pp]*npqy+deltapqtfrontz0[pp]*npqz)*npqy;
            deltapqtfrontz[pp]=deltapqtfrontz0[pp]-(deltapqtfrontx0[pp]*npqx+deltapqtfronty0[pp]*npqy+deltapqtfrontz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqtfrontx[pp]*deltapqtfrontx[pp]+deltapqtfronty[pp]*deltapqtfronty[pp]+deltapqtfrontz[pp]*deltapqtfrontz[pp]);
            deltapqtlen0=sqrt(deltapqtfrontx0[pp]*deltapqtfrontx0[pp]+deltapqtfronty0[pp]*deltapqtfronty0[pp]+deltapqtfrontz0[pp]*deltapqtfrontz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqtfrontx[pp]=deltapqtfrontx[pp]*deltapqtlen0/deltapqtlen;
                deltapqtfronty[pp]=deltapqtfronty[pp]*deltapqtlen0/deltapqtlen;
                deltapqtfrontz[pp]=deltapqtfrontz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqtfrontx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqtfronty[pp]-gammat*upqty;
            Fttestz=-kt*deltapqtfrontz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqtfrontx0[pp]=deltapqtfrontx[pp]+upqtx*dtD;
                deltapqtfronty0[pp]=deltapqtfronty[pp]+upqty*dtD;
                deltapqtfrontz0[pp]=deltapqtfrontz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqtfrontx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqtfronty0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqtfrontz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrfrontx[pp]=deltapqrfrontx0[pp]-(deltapqrfrontx0[pp]*npqx+deltapqrfronty0[pp]*npqy+deltapqrfrontz0[pp]*npqz)*npqx;
            deltapqrfronty[pp]=deltapqrfronty0[pp]-(deltapqrfrontx0[pp]*npqx+deltapqrfronty0[pp]*npqy+deltapqrfrontz0[pp]*npqz)*npqy;
            deltapqrfrontz[pp]=deltapqrfrontz0[pp]-(deltapqrfrontx0[pp]*npqx+deltapqrfronty0[pp]*npqy+deltapqrfrontz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrfrontx[pp]*deltapqrfrontx[pp]+deltapqrfronty[pp]*deltapqrfronty[pp]+deltapqrfrontz[pp]*deltapqrfrontz[pp]);
            deltapqrlen0=sqrt(deltapqrfrontx0[pp]*deltapqrfrontx0[pp]+deltapqrfronty0[pp]*deltapqrfronty0[pp]+deltapqrfrontz0[pp]*deltapqrfrontz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrfrontx[pp]=deltapqrfrontx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrfronty[pp]=deltapqrfronty[pp]*deltapqrlen0/deltapqrlen;
                deltapqrfrontz[pp]=deltapqrfrontz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrfrontx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrfronty[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrfrontz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrfrontx0[pp]=deltapqrfrontx[pp]+upqrx*dtD;
                deltapqrfronty0[pp]=deltapqrfronty[pp]+upqry*dtD;
                deltapqrfrontz0[pp]=deltapqrfrontz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrfrontx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrfronty0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrfrontz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqtfrontx0[pp]=0.0;
            deltapqtfronty0[pp]=0.0;
            deltapqtfrontz0[pp]=0.0;
            deltapqrfrontx0[pp]=0.0;
            deltapqrfronty0[pp]=0.0;
            deltapqrfrontz0[pp]=0.0;
        }
}

void Parboundaryback()
{
    if(TY-1.0-parcy[pp]<=parr[pp])
        {
            dpqx=0.0;
            dpqy=parcy[pp]-(TY-1.0);
            dpqz=0.0;
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqtbackx[pp]=deltapqtbackx0[pp]-(deltapqtbackx0[pp]*npqx+deltapqtbacky0[pp]*npqy+deltapqtbackz0[pp]*npqz)*npqx;
            deltapqtbacky[pp]=deltapqtbacky0[pp]-(deltapqtbackx0[pp]*npqx+deltapqtbacky0[pp]*npqy+deltapqtbackz0[pp]*npqz)*npqy;
            deltapqtbackz[pp]=deltapqtbackz0[pp]-(deltapqtbackx0[pp]*npqx+deltapqtbacky0[pp]*npqy+deltapqtbackz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqtbackx[pp]*deltapqtbackx[pp]+deltapqtbacky[pp]*deltapqtbacky[pp]+deltapqtbackz[pp]*deltapqtbackz[pp]);
            deltapqtlen0=sqrt(deltapqtbackx0[pp]*deltapqtbackx0[pp]+deltapqtbacky0[pp]*deltapqtbacky0[pp]+deltapqtbackz0[pp]*deltapqtbackz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqtbackx[pp]=deltapqtbackx[pp]*deltapqtlen0/deltapqtlen;
                deltapqtbacky[pp]=deltapqtbacky[pp]*deltapqtlen0/deltapqtlen;
                deltapqtbackz[pp]=deltapqtbackz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqtbackx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqtbacky[pp]-gammat*upqty;
            Fttestz=-kt*deltapqtbackz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqtbackx0[pp]=deltapqtbackx[pp]+upqtx*dtD;
                deltapqtbacky0[pp]=deltapqtbacky[pp]+upqty*dtD;
                deltapqtbackz0[pp]=deltapqtbackz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqtbackx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqtbacky0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqtbackz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrbackx[pp]=deltapqrbackx0[pp]-(deltapqrbackx0[pp]*npqx+deltapqrbacky0[pp]*npqy+deltapqrbackz0[pp]*npqz)*npqx;
            deltapqrbacky[pp]=deltapqrbacky0[pp]-(deltapqrbackx0[pp]*npqx+deltapqrbacky0[pp]*npqy+deltapqrbackz0[pp]*npqz)*npqy;
            deltapqrbackz[pp]=deltapqrbackz0[pp]-(deltapqrbackx0[pp]*npqx+deltapqrbacky0[pp]*npqy+deltapqrbackz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrbackx[pp]*deltapqrbackx[pp]+deltapqrbacky[pp]*deltapqrbacky[pp]+deltapqrbackz[pp]*deltapqrbackz[pp]);
            deltapqrlen0=sqrt(deltapqrbackx0[pp]*deltapqrbackx0[pp]+deltapqrbacky0[pp]*deltapqrbacky0[pp]+deltapqrbackz0[pp]*deltapqrbackz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrbackx[pp]=deltapqrbackx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrbacky[pp]=deltapqrbacky[pp]*deltapqrlen0/deltapqrlen;
                deltapqrbackz[pp]=deltapqrbackz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrbackx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrbacky[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrbackz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrbackx0[pp]=deltapqrbackx[pp]+upqrx*dtD;
                deltapqrbacky0[pp]=deltapqrbacky[pp]+upqry*dtD;
                deltapqrbackz0[pp]=deltapqrbackz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrbackx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrbacky0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrbackz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqtbackx0[pp]=0.0;
            deltapqtbacky0[pp]=0.0;
            deltapqtbackz0[pp]=0.0;
            deltapqrbackx0[pp]=0.0;
            deltapqrbacky0[pp]=0.0;
            deltapqrbackz0[pp]=0.0;
        }
}

void Parboundarytop()
{
    if(TZ-1.0-parcz[pp]<=parr[pp])
        {
            dpqx=0.0;
            dpqy=0.0;
            dpqz=parcz[pp]-(TZ-1.0);
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqttopx[pp]=deltapqttopx0[pp]-(deltapqttopx0[pp]*npqx+deltapqttopy0[pp]*npqy+deltapqttopz0[pp]*npqz)*npqx;
            deltapqttopy[pp]=deltapqttopy0[pp]-(deltapqttopx0[pp]*npqx+deltapqttopy0[pp]*npqy+deltapqttopz0[pp]*npqz)*npqy;
            deltapqttopz[pp]=deltapqttopz0[pp]-(deltapqttopx0[pp]*npqx+deltapqttopy0[pp]*npqy+deltapqttopz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqttopx[pp]*deltapqttopx[pp]+deltapqttopy[pp]*deltapqttopy[pp]+deltapqttopz[pp]*deltapqttopz[pp]);
            deltapqtlen0=sqrt(deltapqttopx0[pp]*deltapqttopx0[pp]+deltapqttopy0[pp]*deltapqttopy0[pp]+deltapqttopz0[pp]*deltapqttopz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqttopx[pp]=deltapqttopx[pp]*deltapqtlen0/deltapqtlen;
                deltapqttopy[pp]=deltapqttopy[pp]*deltapqtlen0/deltapqtlen;
                deltapqttopz[pp]=deltapqttopz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqttopx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqttopy[pp]-gammat*upqty;
            Fttestz=-kt*deltapqttopz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqttopx0[pp]=deltapqttopx[pp]+upqtx*dtD;
                deltapqttopy0[pp]=deltapqttopy[pp]+upqty*dtD;
                deltapqttopz0[pp]=deltapqttopz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqttopx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqttopy0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqttopz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrtopx[pp]=deltapqrtopx0[pp]-(deltapqrtopx0[pp]*npqx+deltapqrtopy0[pp]*npqy+deltapqrtopz0[pp]*npqz)*npqx;
            deltapqrtopy[pp]=deltapqrtopy0[pp]-(deltapqrtopx0[pp]*npqx+deltapqrtopy0[pp]*npqy+deltapqrtopz0[pp]*npqz)*npqy;
            deltapqrtopz[pp]=deltapqrtopz0[pp]-(deltapqrtopx0[pp]*npqx+deltapqrtopy0[pp]*npqy+deltapqrtopz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrtopx[pp]*deltapqrtopx[pp]+deltapqrtopy[pp]*deltapqrtopy[pp]+deltapqrtopz[pp]*deltapqrtopz[pp]);
            deltapqrlen0=sqrt(deltapqrtopx0[pp]*deltapqrtopx0[pp]+deltapqrtopy0[pp]*deltapqrtopy0[pp]+deltapqrtopz0[pp]*deltapqrtopz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrtopx[pp]=deltapqrtopx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrtopy[pp]=deltapqrtopy[pp]*deltapqrlen0/deltapqrlen;
                deltapqrtopz[pp]=deltapqrtopz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrtopx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrtopy[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrtopz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrtopx0[pp]=deltapqrtopx[pp]+upqrx*dtD;
                deltapqrtopy0[pp]=deltapqrtopy[pp]+upqry*dtD;
                deltapqrtopz0[pp]=deltapqrtopz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrtopx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrtopy0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrtopz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqttopx0[pp]=0.0;
            deltapqttopy0[pp]=0.0;
            deltapqttopz0[pp]=0.0;
            deltapqrtopx0[pp]=0.0;
            deltapqrtopy0[pp]=0.0;
            deltapqrtopz0[pp]=0.0;
        }
}

void Parboundarybottom()
{
    if(parcz[pp]-1.0<=parr[pp])
        {
            dpqx=0.0;
            dpqy=0.0;
            dpqz=parcz[pp]-1.0;
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqtbottomx[pp]=deltapqtbottomx0[pp]-(deltapqtbottomx0[pp]*npqx+deltapqtbottomy0[pp]*npqy+deltapqtbottomz0[pp]*npqz)*npqx;
            deltapqtbottomy[pp]=deltapqtbottomy0[pp]-(deltapqtbottomx0[pp]*npqx+deltapqtbottomy0[pp]*npqy+deltapqtbottomz0[pp]*npqz)*npqy;
            deltapqtbottomz[pp]=deltapqtbottomz0[pp]-(deltapqtbottomx0[pp]*npqx+deltapqtbottomy0[pp]*npqy+deltapqtbottomz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqtbottomx[pp]*deltapqtbottomx[pp]+deltapqtbottomy[pp]*deltapqtbottomy[pp]+deltapqtbottomz[pp]*deltapqtbottomz[pp]);
            deltapqtlen0=sqrt(deltapqtbottomx0[pp]*deltapqtbottomx0[pp]+deltapqtbottomy0[pp]*deltapqtbottomy0[pp]+deltapqtbottomz0[pp]*deltapqtbottomz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqtbottomx[pp]=deltapqtbottomx[pp]*deltapqtlen0/deltapqtlen;
                deltapqtbottomy[pp]=deltapqtbottomy[pp]*deltapqtlen0/deltapqtlen;
                deltapqtbottomz[pp]=deltapqtbottomz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqtbottomx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqtbottomy[pp]-gammat*upqty;
            Fttestz=-kt*deltapqtbottomz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqtbottomx0[pp]=deltapqtbottomx[pp]+upqtx*dtD;
                deltapqtbottomy0[pp]=deltapqtbottomy[pp]+upqty*dtD;
                deltapqtbottomz0[pp]=deltapqtbottomz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqtbottomx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqtbottomy0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqtbottomz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrbottomx[pp]=deltapqrbottomx0[pp]-(deltapqrbottomx0[pp]*npqx+deltapqrbottomy0[pp]*npqy+deltapqrbottomz0[pp]*npqz)*npqx;
            deltapqrbottomy[pp]=deltapqrbottomy0[pp]-(deltapqrbottomx0[pp]*npqx+deltapqrbottomy0[pp]*npqy+deltapqrbottomz0[pp]*npqz)*npqy;
            deltapqrbottomz[pp]=deltapqrbottomz0[pp]-(deltapqrbottomx0[pp]*npqx+deltapqrbottomy0[pp]*npqy+deltapqrbottomz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrbottomx[pp]*deltapqrbottomx[pp]+deltapqrbottomy[pp]*deltapqrbottomy[pp]+deltapqrbottomz[pp]*deltapqrbottomz[pp]);
            deltapqrlen0=sqrt(deltapqrbottomx0[pp]*deltapqrbottomx0[pp]+deltapqrbottomy0[pp]*deltapqrbottomy0[pp]+deltapqrbottomz0[pp]*deltapqrbottomz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrbottomx[pp]=deltapqrbottomx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrbottomy[pp]=deltapqrbottomy[pp]*deltapqrlen0/deltapqrlen;
                deltapqrbottomz[pp]=deltapqrbottomz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrbottomx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrbottomy[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrbottomz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrbottomx0[pp]=deltapqrbottomx[pp]+upqrx*dtD;
                deltapqrbottomy0[pp]=deltapqrbottomy[pp]+upqry*dtD;
                deltapqrbottomz0[pp]=deltapqrbottomz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrbottomx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrbottomy0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrbottomz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqtbottomx0[pp]=0.0;
            deltapqtbottomy0[pp]=0.0;
            deltapqtbottomz0[pp]=0.0;
            deltapqrbottomx0[pp]=0.0;
            deltapqrbottomy0[pp]=0.0;
            deltapqrbottomz0[pp]=0.0;
        }
}

void Parboundaryleft()
{
    if(parcx[pp]-1.0<=parr[pp])
        {
            dpqx=parcx[pp]-1.0;
            dpqy=0.0;
            dpqz=0.0;
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqtleftx[pp]=deltapqtleftx0[pp]-(deltapqtleftx0[pp]*npqx+deltapqtlefty0[pp]*npqy+deltapqtleftz0[pp]*npqz)*npqx;
            deltapqtlefty[pp]=deltapqtlefty0[pp]-(deltapqtleftx0[pp]*npqx+deltapqtlefty0[pp]*npqy+deltapqtleftz0[pp]*npqz)*npqy;
            deltapqtleftz[pp]=deltapqtleftz0[pp]-(deltapqtleftx0[pp]*npqx+deltapqtlefty0[pp]*npqy+deltapqtleftz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqtleftx[pp]*deltapqtleftx[pp]+deltapqtlefty[pp]*deltapqtlefty[pp]+deltapqtleftz[pp]*deltapqtleftz[pp]);
            deltapqtlen0=sqrt(deltapqtleftx0[pp]*deltapqtleftx0[pp]+deltapqtlefty0[pp]*deltapqtlefty0[pp]+deltapqtleftz0[pp]*deltapqtleftz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqtleftx[pp]=deltapqtleftx[pp]*deltapqtlen0/deltapqtlen;
                deltapqtlefty[pp]=deltapqtlefty[pp]*deltapqtlen0/deltapqtlen;
                deltapqtleftz[pp]=deltapqtleftz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqtleftx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqtlefty[pp]-gammat*upqty;
            Fttestz=-kt*deltapqtleftz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqtleftx0[pp]=deltapqtleftx[pp]+upqtx*dtD;
                deltapqtlefty0[pp]=deltapqtlefty[pp]+upqty*dtD;
                deltapqtleftz0[pp]=deltapqtleftz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqtleftx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqtlefty0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqtleftz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrleftx[pp]=deltapqrleftx0[pp]-(deltapqrleftx0[pp]*npqx+deltapqrlefty0[pp]*npqy+deltapqrleftz0[pp]*npqz)*npqx;
            deltapqrlefty[pp]=deltapqrlefty0[pp]-(deltapqrleftx0[pp]*npqx+deltapqrlefty0[pp]*npqy+deltapqrleftz0[pp]*npqz)*npqy;
            deltapqrleftz[pp]=deltapqrleftz0[pp]-(deltapqrleftx0[pp]*npqx+deltapqrlefty0[pp]*npqy+deltapqrleftz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrleftx[pp]*deltapqrleftx[pp]+deltapqrlefty[pp]*deltapqrlefty[pp]+deltapqrleftz[pp]*deltapqrleftz[pp]);
            deltapqrlen0=sqrt(deltapqrleftx0[pp]*deltapqrleftx0[pp]+deltapqrlefty0[pp]*deltapqrlefty0[pp]+deltapqrleftz0[pp]*deltapqrleftz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrleftx[pp]=deltapqrleftx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrlefty[pp]=deltapqrlefty[pp]*deltapqrlen0/deltapqrlen;
                deltapqrleftz[pp]=deltapqrleftz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrleftx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrlefty[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrleftz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);
            Frtestlen=max(1.0e-12,Frtestlen);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrleftx0[pp]=deltapqrleftx[pp]+upqrx*dtD;
                deltapqrlefty0[pp]=deltapqrlefty[pp]+upqry*dtD;
                deltapqrleftz0[pp]=deltapqrleftz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrleftx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrlefty0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrleftz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqtleftx0[pp]=0.0;
            deltapqtlefty0[pp]=0.0;
            deltapqtleftz0[pp]=0.0;
            deltapqrleftx0[pp]=0.0;
            deltapqrlefty0[pp]=0.0;
            deltapqrleftz0[pp]=0.0;
        }
}

void Parboundaryright()
{
    if(TX-1.0-parcx[pp]<=parr[pp])
        {
            dpqx=parcx[pp]-(TX-1.0);
            dpqy=0.0;
            dpqz=0.0;
            deltapq=sqrt((dpqx)*(dpqx)+(dpqy)*(dpqy)+(dpqz)*(dpqz))-parr[pp];

            Fn=0.0;
            Ftx=0.0;
            Fty=0.0;
            Ftz=0.0;

            npqx=dpqx;
            npqy=dpqy;
            npqz=dpqz;

            npqlen=sqrt(npqx*npqx+npqy*npqy+npqz*npqz);
            if(npqlen>0.0)
            {
                npqx=npqx/npqlen;
                npqy=npqy/npqlen;
                npqz=npqz/npqlen;
            }
            else
            {
                npqx=0.0;
                npqy=0.0;
                npqz=0.0;
            }

            upqx=parux[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqy=paruy[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqz=paruz[pp]-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            upqlen=sqrt(upqx*upqx+upqy*upqy+upqz*upqz);

            upqtx=upqx-(upqx*npqx+upqy*npqy+upqz*npqz)*npqx;
            upqty=upqy-(upqx*npqx+upqy*npqy+upqz*npqz)*npqy;
            upqtz=upqz-(upqx*npqx+upqy*npqy+upqz*npqz)*npqz;

            deltapqtrightx[pp]=deltapqtrightx0[pp]-(deltapqtrightx0[pp]*npqx+deltapqtrighty0[pp]*npqy+deltapqtrightz0[pp]*npqz)*npqx;
            deltapqtrighty[pp]=deltapqtrighty0[pp]-(deltapqtrightx0[pp]*npqx+deltapqtrighty0[pp]*npqy+deltapqtrightz0[pp]*npqz)*npqy;
            deltapqtrightz[pp]=deltapqtrightz0[pp]-(deltapqtrightx0[pp]*npqx+deltapqtrighty0[pp]*npqy+deltapqtrightz0[pp]*npqz)*npqz;

            deltapqtlen=sqrt(deltapqtrightx[pp]*deltapqtrightx[pp]+deltapqtrighty[pp]*deltapqtrighty[pp]+deltapqtrightz[pp]*deltapqtrightz[pp]);
            deltapqtlen0=sqrt(deltapqtrightx0[pp]*deltapqtrightx0[pp]+deltapqtrighty0[pp]*deltapqtrighty0[pp]+deltapqtrightz0[pp]*deltapqtrightz0[pp]);

            if(deltapqtlen>0.0)
            {
                deltapqtrightx[pp]=deltapqtrightx[pp]*deltapqtlen0/deltapqtlen;
                deltapqtrighty[pp]=deltapqtrighty[pp]*deltapqtlen0/deltapqtlen;
                deltapqtrightz[pp]=deltapqtrightz[pp]*deltapqtlen0/deltapqtlen;
            }

            Fn=-kn*deltapq-gamman*((parux[pp])*npqx+(paruy[pp])*npqy+(paruz[pp])*npqz);
            Fttestx=-kt*deltapqtrightx[pp]-gammat*upqtx;
            Fttesty=-kt*deltapqtrighty[pp]-gammat*upqty;
            Fttestz=-kt*deltapqtrightz[pp]-gammat*upqtz;

            if(Fn<0.0) Fn=0.0;

            Fttestlen=sqrt(Fttestx*Fttestx+Fttesty*Fttesty+Fttestz*Fttestz);
            Fttestlen=max(1.0e-12,Fttestlen);

            if(Fttestlen<=muf*Fn)
            {
                deltapqtrightx0[pp]=deltapqtrightx[pp]+upqtx*dtD;
                deltapqtrighty0[pp]=deltapqtrighty[pp]+upqty*dtD;
                deltapqtrightz0[pp]=deltapqtrightz[pp]+upqtz*dtD;

                Ftx=Fttestx;
                Fty=Fttesty;
                Ftz=Fttestz;
            }
            else
            {
                deltapqtrightx0[pp]=-(1.0/kt)*(muf*Fn*Fttestx/Fttestlen+gammat*upqtx);
                deltapqtrighty0[pp]=-(1.0/kt)*(muf*Fn*Fttesty/Fttestlen+gammat*upqty);
                deltapqtrightz0[pp]=-(1.0/kt)*(muf*Fn*Fttestz/Fttestlen+gammat*upqtz);

                Ftx=muf*Fn*Fttestx/Fttestlen;
                Fty=muf*Fn*Fttesty/Fttestlen;
                Ftz=muf*Fn*Fttestz/Fttestlen;
            }

            upqrx=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,0);
            upqry=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,1);
            upqrz=-VCross((parr[pp]+0.5*deltapq)*parwx[pp],(parr[pp]+0.5*deltapq)*parwy[pp],(parr[pp]+0.5*deltapq)*parwz[pp],npqx,npqy,npqz,2);

            deltapqrrightx[pp]=deltapqrrightx0[pp]-(deltapqrrightx0[pp]*npqx+deltapqrrighty0[pp]*npqy+deltapqrrightz0[pp]*npqz)*npqx;
            deltapqrrighty[pp]=deltapqrrighty0[pp]-(deltapqrrightx0[pp]*npqx+deltapqrrighty0[pp]*npqy+deltapqrrightz0[pp]*npqz)*npqy;
            deltapqrrightz[pp]=deltapqrrightz0[pp]-(deltapqrrightx0[pp]*npqx+deltapqrrighty0[pp]*npqy+deltapqrrightz0[pp]*npqz)*npqz;

            deltapqrlen=sqrt(deltapqrrightx[pp]*deltapqrrightx[pp]+deltapqrrighty[pp]*deltapqrrighty[pp]+deltapqrrightz[pp]*deltapqrrightz[pp]);
            deltapqrlen0=sqrt(deltapqrrightx0[pp]*deltapqrrightx0[pp]+deltapqrrighty0[pp]*deltapqrrighty0[pp]+deltapqrrightz0[pp]*deltapqrrightz0[pp]);

            if(deltapqrlen>0.0)
            {
                deltapqrrightx[pp]=deltapqrrightx[pp]*deltapqrlen0/deltapqrlen;
                deltapqrrighty[pp]=deltapqrrighty[pp]*deltapqrlen0/deltapqrlen;
                deltapqrrightz[pp]=deltapqrrightz[pp]*deltapqrlen0/deltapqrlen;
            }

            Frtestx=-krpq*deltapqrrightx[pp]-gammar*upqrx;
            Frtesty=-krpq*deltapqrrighty[pp]-gammar*upqry;
            Frtestz=-krpq*deltapqrrightz[pp]-gammar*upqrz;

            Frtestlen=sqrt(Frtestx*Frtestx+Frtesty*Frtesty+Frtestz*Frtestz);
            Frtestlen=max(1.0e-12,Frtestlen);

            if(Frtestlen<=mur*Fn)
            {
                deltapqrrightx0[pp]=deltapqrrightx[pp]+upqrx*dtD;
                deltapqrrighty0[pp]=deltapqrrighty[pp]+upqry*dtD;
                deltapqrrightz0[pp]=deltapqrrightz[pp]+upqrz*dtD;

                Frx=Frtestx;
                Fry=Frtesty;
                Frz=Frtestz;
            }
            else
            {
                deltapqrrightx0[pp]=-(1.0/krpq)*(mur*Fn*Frtestx/Frtestlen+gammar*upqrx);
                deltapqrrighty0[pp]=-(1.0/krpq)*(mur*Fn*Frtesty/Frtestlen+gammar*upqry);
                deltapqrrightz0[pp]=-(1.0/krpq)*(mur*Fn*Frtestz/Frtestlen+gammar*upqrz);

                Frx=mur*Fn*Frtestx/Frtestlen;
                Fry=mur*Fn*Frtesty/Frtestlen;
                Frz=mur*Fn*Frtestz/Frtestlen;
            }

        Fpqx[pp]=Fpqx[pp]+Fn*npqx+Ftx;
        Fpqy[pp]=Fpqy[pp]+Fn*npqy+Fty;
        Fpqz[pp]=Fpqz[pp]+Fn*npqz+Ftz;

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Ftx, Fty, Ftz, 2);

        Tpqx[pp]=Tpqx[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 0);
        Tpqy[pp]=Tpqy[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 1);
        Tpqz[pp]=Tpqz[pp]+VCross(-(parr[pp]+0.5*deltapq)*npqx,-(parr[pp]+0.5*deltapq)*npqy,-(parr[pp]+0.5*deltapq)*npqz, Frx, Fry, Frz, 2);
        }
        else
        {
            deltapqtrightx0[pp]=0.0;
            deltapqtrighty0[pp]=0.0;
            deltapqtrightz0[pp]=0.0;
            deltapqrrightx0[pp]=0.0;
            deltapqrrighty0[pp]=0.0;
            deltapqrrightz0[pp]=0.0;
        }
}