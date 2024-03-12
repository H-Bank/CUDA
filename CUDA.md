# CUDA
CUDA: GPU Programing

## Alapok
C++ nyelvet használ.

Ezek kellenek, hogy tudjunk GPU-re programot írni.
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
```
## Tárolók

Ezzel lehet a CPU-ra tömböt létrehozni: (CSAK A CPU-n létezik!)
```
int A[] = { 1,2,3,4,5 };
```

Ezzel lehet a GPU-n létrehozni változót, ami a GPU memória területére jön létre. 
```
__device__ int dev_A[5];
```

Ezzel az előtaggal lehet olyan függvényt létrehozni, amit CPU-n és GPU-n is lehet futtatni.
```
__global__ void fg() { }
```
## GPU indexek

Az adott blokkban a száll index lekérése: (Ez csak a blokkos indexet adja meg majd, más szóval ebből lehet több is, mivel több blokkot is futatthatunk majd)
```
int i = threadIdx.x;
```

## Main metódus

CPU-ról GPU-ra másolni az adatok így lehet:
```
cudaMemcpyToSymbol(dev_A, &A, 5*sizeof(int)); 
```
1. dev_A a GPU-s változót tartalmazza (memóriában a helyét)
2. &A a CPU-n az adatot, amit másolunk (lehetne így is: &(A[0]), A)
3. 5*sizeof(int) mekkora az átmásolt adat mérete

A GPU-n fg futtatása:
```
fg <<<1, 5 >>> ();
```
Ahol az 1-es most a blokkot jelenti, hogy 1 blokkot indítunk és az 5-ös pedig azt, hogy mennyi szállat abban.

GPU-ról CPU-ra vissza másolás:
```
cudaMemcpyFromSymbol(A, dev_A, 5 * sizeof(int));
```
1. A a CPU-n az adatot, amit másolunk
2. dev_A a GPU-s változót tartalmazza
3. 5*sizeof(int) mekkora az átmásolt adat mérete

Console-ra kiírás:
```
printf("A[%d]=%d\n", i, A[i]);
```

## GPU-n belüli szorzás kód
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int A[] = { 1,2,3,4,5 };
__device__ int dev_A[5];

int b = 5;
__device__ int dev_b;

__global__ void szorzas(int mennyivel) {
	int i = threadIdx.x;
	dev_A[i] *= mennyivel;
}

int main() {
	cudaMemcpyToSymbol(dev_b, &b, sizeof(int));
	cudaMemcpyToSymbol(dev_A, &(A[0]), 5* sizeof(int));
	szorzas <<<1, 5 >>> (3);
	cudaMemcpyFromSymbol(A, dev_A, 5 * sizeof(int));
	cudaMemcpyFromSymbol(&b, dev_b, sizeof(int));
	for (int i = 0; i < 5; i++)
	{
		printf("A[%d]=%d\n", i, A[i]);
	}
}
```
## Szövegben szó keresés
```
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

std::string szoveg = "ababcab";
std::string szo = "abc";
const int n = (int)szoveg.size();
int m = (int)szo.size();
char szovegTomb[];
__device__ std::string dev_szoveg;
__device__ std::string dev_szo;

int hol = -1;
__device__ int dev_hol;

void keresesCPU()
{
	for (int i = 0; i <= n-m; ++i)
	{
		if (szoveg[i] == szo[0] && szoveg[i + 1] == szo[1] & szoveg[i + 2] == szo[2])
		{
			hol = i;
		}
	}
}

void keresesGPU()
{
	int i = threadIdx.x;
	if (dev_szoveg[i] == dev_szo[0] && dev_szoveg[i + 1] == dev_szo[1] & dev_szoveg[i + 2] == dev_szo[2])
	{
		dev_hol = i;
	}
}

int main() {

	//keresesCPU();

	cudaMemcpyToSymbol(&dev_szoveg, &szoveg, n);
	cudaMemcpyToSymbol(&dev_szo, &szo, m);
	cudaMemcpyToSymbol(&dev_hol, &hol, sizeof(int));
	keresesGPU <<<1, n - m+1 >>> ();
	cudaMemcpyFromSymbol(&szoveg, &dev_szoveg, n);
	cudaMemcpyFromSymbol(&szo, &dev_szo, m);
	cudaMemcpyFromSymbol(&hol, &dev_hol, sizeof(int));
	printf("%d", hol);
}
```
