#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <pymic_kernel.h>


PYMIC_KERNEL
void multiplication(const double *A, const double *B, double *C, const long int *nrows, const long int *ncols)
{
 	for(int i = 0; i < *nrows; i++)
	{
    	for(int j = 0; j < *ncols; j++)
        {
        	for(int k = 0; k < *ncols; k++)
			{
	            *(C+i*10+j) += *(A+i*10+k) * *(B+k*10+j);
			}	
        }
    }
}

/*
const int N = 244;
void main()
{
	int i, j;
    int a[N][N];
    int b[N][N];
    int c[N][N];

    srand(time(0));
    omp_set_num_threads(N);
    #pragma omp for
	for(i = 0; i < N; i++)
	{ 
		for(j = 0; j < N; j++)
		{
			a[i][j] = rand();
			b[i][j] = rand();
		}
	 }
	 
	#pragma omp for
   	for(i = 0; i < N; i++)
    {
    	for(j = 0; j < N; j++)
    	{
    		for(int k = 0; k < N; k++)
			{
	            c[i][j] += a[i][k] * b[k][j];
			}
    	}
    }
    
    printf("The product of matrix a and matrix b is: \n");
    for(i = 0; i < N; i++)
    {
    	for(j = 0; j < N; j++)
    	{
    		printf("%d\t",c[i][j]); 
    	}
    	printf("\n");
    }
}*/
