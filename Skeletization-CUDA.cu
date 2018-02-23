#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#ifdef TIME
  #define COMM 1
#elif NOTIME 
  #define COMM 0
#endif

#define MASK_WIDTH 5
#define TILE_WIDTH 32
#define GPU 1
#define COMMENT "skeletization_GPU"
#define RGB_COMPONENT_COLOR 255


typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

typedef struct {
    int x, y;
} Par;

double time_total;
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

void RGBtoGrayScaleImage(PPMImage *image, int *GrayScale) {

	int i;
	int cols;

	int n = image->y * image->x;

	cols = image->x;

	for (i = 0; i < n; i++) {
		GrayScale[(i/cols)*cols+(i%cols)]=(int)(0.2126*image->data[i].red+0.7152*image->data[i].green+0.0722 *image->data[i].blue);
	}
}

void Histogramify(int *GrayScale, int *histogram, int rows, int cols)
{
    int i,j;
    for(i=0; i<256; i++) histogram[i]=0;

    for(i=0; i<rows; i++)
        for(j=0; j<cols; j++)
            histogram[GrayScale[i*cols + j]]+=1;
}

int Otsu(int * histogram, int size)
{
    int i,total=size;
    float sum=0;
    for(i=0; i< 256; i++) sum+=i*histogram[i];
    
    float sumB=0;
    int wB=0;
    int wF=0;
    
    float varMax=0;
    int threshold=0;
    
    for(i=0; i<256; i++)
    {
        wB+=histogram[i];
        if(wB==0) continue;
        
        wF=total-wB;
        if(wF==0) break;
        
        sumB+=(float)(i*histogram[i]);
        float mB=sumB/wB;
        float mF=(sum-sumB)/wF;

        float varBetween=(float)wB*(float)wF*(mB-mF)*(mB-mF);
        
        if(varBetween>varMax)
        {
            varMax=varBetween;
            threshold=i;
        }
    }
    return threshold;
}   

void Neighbours(int x, int y, int **GrayScale, int *neighbours)
{
    int i,total=0;
    int X_index[8]={-1,-1,0,1,1,1,0,-1};
    int Y_index[8]={0,1,1,1,0,-1,-1,-1};
    for(i=0; i<8; i++)
    {
        neighbours[i]=GrayScale[x+X_index[i]][y+Y_index[i]];
        total+=neighbours[i]; 
    }
    neighbours[8]=total;
}

int transitions(int *neighbours)
{ 
    int i,ans=0;
    for(i=0; i<7; i++)
        if(neighbours[i]==0 && neighbours[i+1]==1) ans+=1;
    if(neighbours[7]==0 && neighbours[0]==1) ans+=1;
    return ans;
}

__global__ void Update(int *GrayScale_, int *d_changing1, int linhas, int colunas)
{
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int fil = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int index = fil * colunas + col;

	if (fil>0 && col>0 && fil < linhas-1 && col < colunas-1)
	{
		if(d_changing1[index]==1)
			GrayScale_[index]=0;
	}
}
__global__ void Condition(int *GrayScale_, int *d_changing1, int *cont, int linhas, int colunas, int flag)
{
	int X_index[8]={-1,-1,0,1,1,1,0,-1};
	int Y_index[8]={0,1,1,1,0,-1,-1,-1};
	int neighbours[9]={0,0,0,0,0,0,0,0,0};
	int i,j,total=0;
	int ans=0;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int fil = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int index = fil * colunas + col;
	if (fil>0 && col>0 && fil < linhas-1 && col < colunas-1)
	{
		d_changing1[index]=0;
		for(i=0; i<8; i++)
		{
		    neighbours[i]=GrayScale_[(fil+X_index[i])*colunas + (col+Y_index[i])];
		    total+=neighbours[i]; 
		}
		
		neighbours[8]=total;
		for(j=0; j<7; j++)
		{
	        if(neighbours[j]==0 && neighbours[j+1]==1) 
				ans=ans+1;
		}

	    if(neighbours[7]==0 && neighbours[0]==1) 
			ans=ans+1;
		if(flag!=1)
		{
			if(GrayScale_[fil*colunas + col]==1 && neighbours[8]>=2 && neighbours[8]<=6 
				&& ans==1 && neighbours[0]*neighbours[2]*neighbours[4]==0 
				&& neighbours[2]*neighbours[4]*neighbours[6]==0)
			{
				d_changing1[index]=1;
				cont[flag]=1;
			}
		}
		else
		{
			if(GrayScale_[fil*colunas + col]==1 && neighbours[8]>=2 && neighbours[8]<=6 
				&& ans==1 && neighbours[0]*neighbours[2]*neighbours[6]==0 
				&& neighbours[0]*neighbours[4]*neighbours[6]==0)
			{
				d_changing1[index]=1;
				cont[flag]=1;
			}
		}
	}
}

void zhangsuen_CUDA(int *GrayScale, int rows, int cols)
{
	double t_start, t_end;

	//	execution time-CUDA
	double create_buffer,offload_send,kernel,offload_receive;
	create_buffer=0;
	offload_send=0;
	kernel=offload_receive=0;
	time_total=0;

    int *changing1=(int*) malloc(rows * cols *sizeof(int));
	int *d_changing1=(int*) malloc(rows * cols *sizeof(int));
	int *d_cont=(int*) malloc(2*sizeof(int));
	int *cont=(int*) malloc(2*sizeof(int));
	int *d_GrayScale;

	int sizePixel=(int) (rows * cols * sizeof(int));
	int sizeC=(int) (rows * cols *sizeof(int)); 
	int size=(int) (2*sizeof(int));

	t_start = rtclock();
	cudaMalloc(&d_changing1, sizeC);
	cudaMalloc(&d_GrayScale, sizePixel);
	cudaMalloc(&d_cont,size);
	t_end = rtclock();
	create_buffer=create_buffer+(t_end - t_start);
	
	//  define number of blocks and threads
	int par1=(int)ceil((float)cols / TILE_WIDTH);
	int par2=(int) ceil((float)rows / TILE_WIDTH);
		
	dim3 dimGrid(par1,par2,1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	*changing1=0;
	cont[0]=cont[1]=1;
	while(cont[0]>0 || cont[1]>0)
    {  
		cont[0]=0;
		cont[1]=0;
		/*=========================== First Condition ================================*/		
		//  copy memory to the GPU here
		t_start = rtclock();
		cudaMemcpy(d_GrayScale,GrayScale,sizePixel,cudaMemcpyHostToDevice);
		cudaMemcpy(d_changing1,changing1,sizeC,cudaMemcpyHostToDevice);
		cudaMemcpy(d_cont,cont,size,cudaMemcpyHostToDevice);
		t_end = rtclock();
		offload_send=offload_send+(t_end - t_start);
		
		//  launch the GPU Kernel here
		t_start = rtclock();
		Condition<<<dimGrid,dimBlock>>>(d_GrayScale,d_changing1,d_cont,rows,cols,0);
		cudaDeviceSynchronize();
		t_end = rtclock();
		kernel=kernel+(t_end - t_start);

		//  copy the GPU memory back to the CPU here   
		t_start = rtclock();
		cudaMemcpy(changing1, d_changing1, sizeC, cudaMemcpyDeviceToHost);
		cudaMemcpy(cont, d_cont,size, cudaMemcpyDeviceToHost);
		t_end = rtclock();
		offload_receive=offload_receive+(t_end - t_start);

		/*============ First Update ============*/
		//  copy memory to the GPU here
		t_start = rtclock();
		cudaMemcpy(d_GrayScale,GrayScale,sizePixel,cudaMemcpyHostToDevice);
		cudaMemcpy(d_changing1,changing1,sizeC,cudaMemcpyHostToDevice);
		t_end = rtclock();
		offload_send=offload_send+(t_end - t_start);

		//  launch the GPU Kernel here
		t_start = rtclock();
		Update<<<dimGrid,dimBlock>>>(d_GrayScale,d_changing1,rows,cols);
		cudaDeviceSynchronize();
		t_end = rtclock();
		kernel=kernel+(t_end - t_start);

		//  copy the GPU memory back to the CPU here   
		t_start = rtclock();
		cudaMemcpy(GrayScale, d_GrayScale, sizePixel, cudaMemcpyDeviceToHost);
		t_end = rtclock();
		offload_receive=offload_receive+(t_end - t_start);
		
		cont[1]=0;
		/*========================== Second Condition ===========================*/		
		//  copy memory to the GPU here
		t_start = rtclock();
		cudaMemcpy(d_GrayScale,GrayScale,sizePixel,cudaMemcpyHostToDevice);
		cudaMemcpy(d_changing1,changing1,sizeC,cudaMemcpyHostToDevice);
		cudaMemcpy(d_cont,cont,size,cudaMemcpyHostToDevice);
		t_end = rtclock();
		offload_send=offload_send+(t_end - t_start);
		
		//  launch the GPU Kernel here
		t_start = rtclock();
		Condition<<<dimGrid,dimBlock>>>(d_GrayScale,d_changing1,d_cont,rows,cols,1);
		cudaDeviceSynchronize();
		t_end = rtclock();
		kernel=kernel+(t_end - t_start);

		//  copy the GPU memory back to the CPU here   
		t_start = rtclock();
		cudaMemcpy(changing1, d_changing1, sizeC, cudaMemcpyDeviceToHost);
		cudaMemcpy(cont, d_cont,size, cudaMemcpyDeviceToHost);
		t_end = rtclock();
		offload_receive=offload_receive+(t_end - t_start);

		/*============ First Update ============*/
		//  copy memory to the GPU here
		t_start = rtclock();
		cudaMemcpy(d_GrayScale,GrayScale,sizePixel,cudaMemcpyHostToDevice);
		cudaMemcpy(d_changing1,changing1,sizeC,cudaMemcpyHostToDevice);
		t_end = rtclock();
		offload_send=offload_send+(t_end - t_start);

		//  launch the GPU Kernel here
		t_start = rtclock();
		Update<<<dimGrid,dimBlock>>>(d_GrayScale,d_changing1,rows,cols);
		cudaDeviceSynchronize();
		t_end = rtclock();
		kernel=kernel+(t_end - t_start);

		//  copy the GPU memory back to the CPU here   
		t_start = rtclock();
		cudaMemcpy(GrayScale, d_GrayScale, sizePixel, cudaMemcpyDeviceToHost);
		t_end = rtclock();
		offload_receive=offload_receive+(t_end - t_start);
    }
	time_total=create_buffer+offload_send+kernel+offload_receive;
	if(COMM==1)
	{
		printf("Parallel time : %0.6lfs \n", time_total);
		printf("	create buffer : %0.6lfs \n", create_buffer);
		printf("	offload send : %0.6lfs \n",offload_send);
		printf("	kernel : %0.6lfs \n", kernel);
		printf("	offload receive : %0.6lfs \n", offload_receive);
	}
	cudaFree(d_GrayScale);
	cudaFree(d_changing1);
}

void zhangsuen(int *GrayScale, int rows, int cols)
{
    int *neighbours=(int *) malloc(9*sizeof (int));
    int *changing1=(int*) malloc(rows * cols *sizeof(int));
	int *GrayScale_ = (int*) malloc(rows*cols * sizeof(int));
    int i,total,cont1=1;
    int j,k,cont2=1;
	int ans=0;
    int X_index[8]={-1,-1,0,1,1,1,0,-1};
    int Y_index[8]={0,1,1,1,0,-1,-1,-1};
    while(cont1>0 || cont2>0)
    {
        cont1=0;
        cont2=0;
        for(i=1; i<rows-1; i++)
        {   
            for(j=1; j<cols-1; j++)
            {       
					total=0;
					ans=0;
					changing1[i*cols+j]=0;
					for(k=0; k<8; k++)
					{
						neighbours[k]=GrayScale[(i+X_index[k])*cols + (j+Y_index[k])];
						total+=neighbours[k]; 
					}
					neighbours[8]=total;

		            for(k=0; k<7; k++)
					{
						if(neighbours[k]==0 && neighbours[k+1]==1) 
							ans=ans+1;
					}

					if(neighbours[7]==0 && neighbours[0]==1) 
						ans=ans+1;

					if(GrayScale[i*cols+j]==1 && neighbours[8]>=2 && neighbours[8]<=6 && ans==1 && neighbours[0]*neighbours[2]*neighbours[4]==0 && neighbours[2]*neighbours[4]*neighbours[6]==0)
					{
						changing1[i*cols+j]=1;
					}
            }
        }

        for(i=1; i<rows-1; i++)
		{
		    for(j=1; j<cols-1; j++)
			{
				if(changing1[i*cols +j]==1){
					cont1=cont1+1;
		        	GrayScale[i*cols+j]=0;
				}
			}
		}
        
        for(i=1; i<rows-1; i++)
        {
            for(j=1; j<cols-1; j++)
            {
					total=0;
					ans=0;
					changing1[i*cols+j]=0;
					for(k=0; k<8; k++)
					{
						neighbours[k]=GrayScale[(i+X_index[k])*cols + (j+Y_index[k])];
						total+=neighbours[k]; 
					}
					neighbours[8]=total;

		            for(k=0; k<7; k++)
					{
						if(neighbours[k]==0 && neighbours[k+1]==1) 
							ans=ans+1;
					}

					if(neighbours[7]==0 && neighbours[0]==1) 
						ans=ans+1;

					if(GrayScale[i*cols+j]==1 && neighbours[8]>=2 && neighbours[8]<=6 && ans==1 && neighbours[0]*neighbours[2]*neighbours[6]==0 && neighbours[0]*neighbours[4]*neighbours[6]==0)
					{
						changing1[i*cols+j]=1;
					}
            }
        }
        for(i=1; i<rows-1; i++)
		{
		    for(j=1; j<cols-1; j++)
			{
				if(changing1[i*cols +j]==1){
					cont2=cont2+1;
		        	GrayScale[i*cols+j]=0;
				}
			}
		}
	
    }
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

	double t_start, t_end;
    int rows, cols,i,j;
	if(COMM==1)
		printf("====== %s ======\n",argv[1]);
	// read image
	PPMImage *image = readPPM(argv[1]);
	cols = image->x;
	rows = image->y;

	int *histogram= (int *) malloc(sizeof (int) *256);
	int *GrayScale = (int*) malloc(image->x * image->y * sizeof(int));
	int *GrayScale2 = (int*) malloc(image->x * image->y * sizeof(int));

	//	convert image from RGG to Gray scale
	RGBtoGrayScaleImage(image,GrayScale);	

	//	histogram of the gray scale image
	Histogramify(GrayScale,histogram, rows, cols);

	int var_otsu=Otsu(histogram,rows*cols);
		
	//	convert image from gray scale to binary image
	for(i=0; i<rows; i++)
		   for(j=0; j<cols; j++)
		       if(GrayScale[i*cols + j]<var_otsu) GrayScale[i*cols + j]=GrayScale2[i*cols + j]=1;
		       else GrayScale[i*cols + j]=GrayScale2[i*cols + j]=0;
	//	skeletonize binary image
	t_start = rtclock();
	zhangsuen(GrayScale,rows,cols);
	t_end = rtclock();

	if(COMM==1)
		fprintf(stdout, "\nSerial time: %0.6lfs\n", t_end - t_start); 

	zhangsuen_CUDA(GrayScale2,rows,cols);

	if(COMM==1)
		fprintf(stdout, "Speedup: %0.3lf\n\n",(double)(t_end - t_start)/(double)time_total); 

	for(i=0; i<rows; i++)
	{
		for(j=0; j<cols; j++)
		{
		   if(GrayScale[i*cols + j]==1)
		   {
		       image->data[i*cols+j].red=255;
		       image->data[i*cols+j].green=255;
		       image->data[i*cols+j].blue=255;
		   } 
		   else
		   {
		       image->data[i*cols+j].red=0;
		       image->data[i*cols+j].green=0;
		       image->data[i*cols+j].blue=0;
		   }       
		}
	}
	if(COMM==0)
	{
		writePPM(image);
		free(image);
	}

}
