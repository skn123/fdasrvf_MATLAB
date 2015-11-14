
#include "ElasticCurvesReparam.h"

#if !defined(MATLAB_MEX_FILE) && defined(TESTELASTICCURVESRO)

void main()
{
	return();
}
#endif

#ifdef MATLAB_MEX_FILE

#define ELASTICCURVESREPARAM

std::map<integer *, integer> *CheckMemoryDeleted;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 9)
	{
		mexErrMsgTxt("[opt,swap,fopts,comtime] = ElasticCurvesReparam(C1, C2, w, onlyDP, rotated, isclosed, skipm, method, autoselectC).\n");
	}
	double *C1, *C2;
	double w = 0;
	C1 = mxGetPr(prhs[0]);
	C2 = mxGetPr(prhs[1]);
	/* dimensions of input matrices */
	integer d, n, rotated, isclosed, onlyDP, skipm, autoselectC;
	n = mxGetM(prhs[0]);
	d = mxGetN(prhs[0]);

	//std::cout << "(n, d):" << n << "," << d << std::endl;

	if (mxGetM(prhs[1]) != n || mxGetN(prhs[1]) != d)
	{
		mexErrMsgTxt("The size of matrix C2 does not match the size of C1.\n");
	}
	w = mxGetScalar(prhs[2]);
	onlyDP = static_cast<integer> (mxGetScalar(prhs[3]));
	rotated = static_cast<integer> (mxGetScalar(prhs[4]));
	isclosed = static_cast<integer> (mxGetScalar(prhs[5]));
	skipm = static_cast<integer> (mxGetScalar(prhs[6]));
	char methodname[30] = "";
	mxGetString(prhs[7], methodname, 30);
	autoselectC = static_cast<integer> (mxGetScalar(prhs[8]));

	init_genrand(0);

	CheckMemoryDeleted = new std::map<integer *, integer>;

	integer numofmanis = 3;
    integer numofmani1 = 1;
    integer numofmani2 = 1;
    integer numofmani3 = 1;
    L2SphereVariable FNSV(n);
    OrthGroupVariable OGV(d);
    EucVariable EucV(1);
    ProductElement *Xopt = new ProductElement(numofmanis, &FNSV, numofmani1, &OGV, numofmani2, &EucV, numofmani3);

    bool swap;
	plhs[2] = mxCreateDoubleMatrix(5, 1, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(5, 1, mxREAL);
	double *fopts = mxGetPr(plhs[2]), *comtime = mxGetPr(plhs[3]);
	integer ns, lms;

	DriverElasticCurvesRO(C1, C2, d, n, w, rotated != 0, isclosed != 0, onlyDP != 0, skipm, methodname,
		autoselectC, Xopt, swap, fopts, comtime, ns, lms);

	/*create output matrix*/
	integer sizex = n + d * d + 1;
	plhs[0] = mxCreateDoubleMatrix(sizex, 1, mxREAL);
	double *opt = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleScalar(static_cast<double> (swap));

	const double *Xoptptr = Xopt->ObtainReadData();
	integer inc = 1;
	dcopy_(&sizex, const_cast<double *> (Xoptptr), &inc, opt, &inc);

	delete Xopt;

	std::map<integer *, integer>::iterator iter = CheckMemoryDeleted->begin();
	for (iter = CheckMemoryDeleted->begin(); iter != CheckMemoryDeleted->end(); iter++)
	{
		if (iter->second != 1)
			std::cout << "Global address:" << iter->first << ", sharedtimes:" << iter->second << std::endl;
	}
	delete CheckMemoryDeleted;
	return;
}

#endif
