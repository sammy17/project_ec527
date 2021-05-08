#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define RED  "\x1B[31m"
#define RESET  "\x1B[0m"

//#define USE_RANDOM

#ifdef USE_RANDOM
#define INPUT_DIMENSIONS 1000
#else
#define INPUT_DIMENSIONS 28
#endif

// #define INPUT_DIMENSIONS 28
#define C1_LENGTH 6
#define C1_DIMENSIONS 24
#define S1_LENGTH 6
#define S1_DIMENSIONS 12
#define C2_LENGTH 12
#define C2_DIMENSIONS 8
#define S2_LENGTH 12
#define S2_DIMENSIONS 4
#define OUTPUT_LENGTH 10
#define KERNEL_SIZE 5
#define UPPER 1
#define LOWER -1
#define BATCH_SIZE 200
#define LEARNING_RATE 0.01

// struct VECTOR{
//     double* vector;
// };
//typedef struct VECTOR vector;
typedef double* vector;

vector new_vector(int length){
    vector new = malloc(sizeof(double)*length);
    //new->vector = malloc(sizeof(double)*length);
    return new;
}

// struct ARRAY{
//     double** matrix;
// };
//typedef struct ARRAY array;
typedef double** array;

array new_array(int rows, int columns){
    //array* new = malloc(sizeof(array*));
    array new = malloc(sizeof(double*)*rows);
    int i;
    for(i = 0; i < rows; i++){
        new[i] = malloc(sizeof(double)*columns);
    }
    return new;
}

// struct KERNEL_ARRAY{
//     array** kernels;
// };
// typedef struct KERNEL_ARRAY kernel_array;
typedef array** kernel_array;

kernel_array new_kernel_array(int rows, int columns, int kernel_dimensions){
    kernel_array new = malloc(sizeof(array**)*rows); //malloc(sizeof(kernel_array*)); //
    // new->kernels = malloc(sizeof(array**)*rows);
    int j,k;
    for(j = 0; j < rows; j++){
        new[j] = malloc(sizeof(array*)*columns);
    }
    for(j = 0; j < rows; j++){
        for(k = 0; k < columns; k++){
            new[j][k] = new_array(kernel_dimensions,kernel_dimensions);
        }
    }
    return new;
}

struct IMAGE_VECTOR{
    array* image;
};
typedef struct IMAGE_VECTOR image_vector;

image_vector* new_image_vector(int image_dimensions, int length){
    image_vector* new = malloc(sizeof(image_vector*));
    new->image = malloc(sizeof(array*)*length);
    int i;
    for(i = 0; i < length; i++){
        new->image[i] = new_array(image_dimensions, image_dimensions);
    }
    return new;
}

struct CNN{
    array input_image;
    kernel_array C1_Kernels;
    vector C1_Biases;
    image_vector* C1_Images;
    image_vector* S1_Images;
    kernel_array C2_Kernels;
    vector C2_Biases;
    image_vector* C2_Images;
    image_vector* S2_Images;
    vector S2_Vector;
    array output_weights;
    vector output_biases;
    vector calculated_output;
    vector desired_output;
};
typedef struct CNN cnn;

cnn* new_cnn(){
    cnn* new = malloc(sizeof(cnn));
    // new->input_image = malloc(sizeof(array*));
    // new->C1_Kernels = malloc(sizeof(kernel_array*));
    //new->C1_Biases = malloc(sizeof(vector*));
    // new->C1_Images = malloc(sizeof(image_vector)); chath
    // new->S1_Images = malloc(sizeof(image_vector));
    // new->C2_Kernels = malloc(sizeof(kernel_array*));
    //new->C2_Biases = malloc(sizeof(vector*));
    // new->C2_Images = malloc(sizeof(image_vector*)); chath
    // new->S2_Images = malloc(sizeof(image_vector*));
    //new->S2_Vector = malloc(sizeof(vector*));
    // new->output_weights = malloc(sizeof(array*));
    //new->output_biases = malloc(sizeof(vector*));
    //new->calculated_output = malloc(sizeof(vector*));
    //new->desired_output = malloc(sizeof(vector*));
    new->input_image = new_array(INPUT_DIMENSIONS, INPUT_DIMENSIONS);
    new->C1_Kernels = new_kernel_array(1, C1_LENGTH, KERNEL_SIZE);
    new->C1_Biases = new_vector(C1_LENGTH);
    new->C1_Images = new_image_vector(C1_DIMENSIONS, C1_LENGTH);
    new->S1_Images = new_image_vector(S1_DIMENSIONS, S1_LENGTH);
    new->C2_Kernels = new_kernel_array(S1_LENGTH, C2_LENGTH, KERNEL_SIZE);
    new->C2_Biases = new_vector(C2_LENGTH);
    new->C2_Images = new_image_vector(C2_DIMENSIONS, C2_LENGTH);
    new->S2_Images = new_image_vector(S2_DIMENSIONS, S2_LENGTH);
    new->S2_Vector = new_vector(S2_DIMENSIONS * S2_DIMENSIONS * S2_LENGTH);
    new->output_weights = new_array(OUTPUT_LENGTH, S2_DIMENSIONS * S2_DIMENSIONS * S2_LENGTH);
    new->output_biases = new_vector(OUTPUT_LENGTH);
    new->calculated_output = new_vector(OUTPUT_LENGTH);
    new->desired_output = new_vector(OUTPUT_LENGTH);
    return new;
}

void load_kernels(cnn* network);
void load_network(cnn* network);
void load_weights(cnn* network);
void load_biases(cnn* network);
void load_output(cnn* network);
double uniform_rand();
void zero_activations(cnn* network);
void zero_parameters(cnn* network);
void zero_network(cnn* network);
void read_headers(FILE *image_ptr, FILE *label_ptr);
void load_image(cnn* network, FILE *image_ptr, FILE *label_ptr);
void load_C1(cnn* network);
void load_S1(cnn* network);
void load_C2(cnn* network);
void load_S2(cnn* network);
void concatenate_S2(cnn* network);
void load_output(cnn* network);
double dot_product(array array1, array array2, int dimensions);
double activation(double x);
double activation_derivative(double x);
void free_array(array array, int rows);
void free_image_vector(image_vector* images, int dimensions, int length);
double average_matrix(array image, int dimensions);
void backpropagation(cnn* network, cnn* gradient);
void forward_propagate(cnn* network);
void gradient_descent(cnn* network, cnn* gradient);
double loss_function(cnn* network);
void print_output(cnn* network);
void update_batch_gradient(cnn* image_gradient, cnn* batch_gradient);
void print_image(array image, int dimensions);
double output_activation(double x);
double output_activation_derivative(double x);
void save_parameters(cnn* network);
void save_kernels(cnn* network);
void save_weights(cnn* network);
void save_biases(cnn* network);

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

double fRand(void)
{
  double f = (double)random() / (double)(RAND_MAX);
  return f;
}

int main(int argc, char** argv) {
    FILE *image_ptr;
    FILE *label_ptr;
    srand(time(NULL));
    struct timespec time_start, time_stop;
    struct timespec time_start1, time_stop1;
    double bp_time, total_time;
    cnn* network = new_cnn();
    cnn* batch_gradient = new_cnn();
    cnn* image_gradient = new_cnn();
    
    zero_network(batch_gradient);
    zero_network(image_gradient);
    zero_network(network);
    load_network(network);
    int n, i, j, m, N, M;
    N = 1; M = 1; // M = epoch, N = iterations
    double time_stamp[M][N][BATCH_SIZE];
    
    //Training Loop
    double batch_loss = 0;
    double iteration_loss = 0;
    clock_gettime(CLOCK_REALTIME, &time_start1);
    for(m = 0; m < M; m++){ // EPOCH
        image_ptr = fopen("Images.idx3-ubyte","r");
        label_ptr = fopen("Labels.idx1-ubyte", "r");
        read_headers(image_ptr, label_ptr);
        for(n = 0; n < N; n++){ // Images per batch = (60000)/BATCH_SIZE //(double)(60000)/BATCH_SIZE - N was 60000/BATCH_SIZE
            zero_network(batch_gradient);
            for(i = 0; i < BATCH_SIZE; i++){
                zero_network(image_gradient);
                zero_activations(network);
                load_image(network, image_ptr, label_ptr);
                forward_propagate(network);
                clock_gettime(CLOCK_REALTIME, &time_start);
                backpropagation(network, image_gradient);
                clock_gettime(CLOCK_REALTIME, &time_stop);
                time_stamp[m][n][i] = interval(time_start, time_stop);
                update_batch_gradient(image_gradient, batch_gradient);
                batch_loss += loss_function(network);
            }
            gradient_descent(network, batch_gradient);
            printf("Loss for batch(%d, %d): %lf\n",m, n, batch_loss/BATCH_SIZE);
            iteration_loss += batch_loss;
            batch_loss = 0;
            save_parameters(network);
        }
        printf("Loss for iteration(%d): %lf\n",m,iteration_loss/60000);
        iteration_loss = 0;
        fclose(image_ptr);
        fclose(label_ptr);   
    }
    clock_gettime(CLOCK_REALTIME, &time_stop1);
    for(m = 0; m < M; m++){
        for(n = 0; n < N; n++){
            for(i = 0; i < BATCH_SIZE; i++){
                bp_time += time_stamp[m][n][i];
            }
        }
    }
    printf("Average time for backpropagation over %d iterations = %lf s\n",M*N*BATCH_SIZE, (bp_time/(M*N*BATCH_SIZE)));
    printf("Total time for backpropagation = %lf s\n",bp_time);
    total_time = interval(time_start1, time_stop1);
    printf("Total elapsed time = %lf s\n",total_time);

    return (EXIT_SUCCESS);
}

void save_parameters(cnn* network){
    save_kernels(network);
    save_weights(network);
    save_biases(network);
}

void save_kernels(cnn* network){
    FILE *kernels_ptr = fopen("kernels","w");
    int i, n, j, k;
    
    for(i = 0; i < C1_LENGTH; i++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                fprintf(kernels_ptr,"%lf ",network->C1_Kernels[0][i][j][k]);
            }
        }
    }
    
    for(i = 0; i < S1_LENGTH; i++){
        for(n = 0; n < C2_LENGTH; n++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    fprintf(kernels_ptr,"%lf ",network->C2_Kernels[i][n][j][k]);
                }
            }
        }
    }
    fclose(kernels_ptr);
}

void save_weights(cnn* network){
    FILE *weight_ptr = fopen("weights","w");
    int j, k;
    for(j = 0; j < OUTPUT_LENGTH; j++){
        for(k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++){
            fprintf(weight_ptr,"%lf ",network->output_weights[j][k]);
        }
    }
    fclose(weight_ptr);
}

void save_biases(cnn* network){
    FILE *bias_ptr = fopen("biases","w");
    int i;
    for(i = 0; i < C1_LENGTH; i++){
        fprintf(bias_ptr,"%lf ",network->C1_Biases[i]);
    }
    for(i = 0; i < C2_LENGTH; i++){
        fprintf(bias_ptr,"%lf ",network->C2_Biases[i]);
    }
    for(i = 0; i < OUTPUT_LENGTH; i++){
        fprintf(bias_ptr,"%lf ",network->output_biases[i]);
    }
    fclose(bias_ptr);
}

void print_output(cnn* network){
    int i;
    for(i = 0; i < OUTPUT_LENGTH; i++){
        printf("%lf == %lf\n",network->calculated_output[i],network->desired_output[i]);
    }
    printf("\n");
}

void forward_propagate(cnn* network){
    load_C1(network);
    load_S1(network);
    load_C2(network);
    load_S2(network);
    concatenate_S2(network);
    load_output(network);
}

//Loads the network's weights, kernels, and biases
void load_network(cnn* network){
    load_kernels(network);
    load_weights(network);
    load_biases(network);
}

//Loads a uniform distribution of values for the kernels in the network
void load_kernels(cnn* network){
    int i, n, j, k;
    FILE *kernel_ptr = fopen("kernels","r");
    double value = 0;
    //C1 will have C1_LENGTH, KERNEL_SIZE x KERNEL_SIZE kernels
    for(i = 0; i < C1_LENGTH; i++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                fscanf(kernel_ptr,"%lf",&value);
                network->C1_Kernels[0][i][j][k] = value;
            }
        }
        //print_image(network->C1_Kernels->kernels[0][i], KERNEL_SIZE);
    }
    
    //C2 will have C2_LENGTH, KERNEL_SIZE x KERNEL_SIZE kernels
    for(i = 0; i < S1_LENGTH; i++){
        for(n = 0; n < C2_LENGTH; n++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    fscanf(kernel_ptr,"%lf",&value);
                    network->C2_Kernels[i][n][j][k] = value;
                }
            }
            //print_image(network->C2_Kernels->kernels[i][n], KERNEL_SIZE);
        }
    }
    fclose(kernel_ptr);
}

//Loads a uniform distribution of values for the weights in the network
void load_weights(cnn* network){
    FILE *weight_ptr = fopen("weights","r");
    int j, k;
    double value = 0;
    for(j = 0; j < OUTPUT_LENGTH; j++){
        for(k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++){
            fscanf(weight_ptr,"%lf",&value);
            network->output_weights[j][k] = value;
        }
    }
    fclose(weight_ptr);
}

//Sets all biases to 0
void load_biases(cnn* network){
    int i;
    FILE *biases_ptr = fopen("biases","r");
    double value = 0;
    for(i = 0; i < C1_LENGTH; i++){
        fscanf(biases_ptr,"%lf",&value);
        network->C1_Biases[i] = value;
    }
    for(i = 0; i < C2_LENGTH; i++){
        fscanf(biases_ptr,"%lf",&value);
        network->C2_Biases[i] = value;
    }
    for(i = 0; i < OUTPUT_LENGTH; i++){
        fscanf(biases_ptr,"%lf",&value);
        network->output_biases[i] = value;
    }
    fclose(biases_ptr);
}

//Sets all values within a network to 0
void zero_network(cnn* network){
    zero_activations(network);
    zero_parameters(network);
}

//Sets all activations within a network to 0
void zero_activations(cnn* network){
    int i, j, k;
    for(i = 0; i < C1_LENGTH; i++){
        for(j = 0; j < C1_DIMENSIONS; j++){
            for(k = 0; k < C1_DIMENSIONS; k++){
                network->C1_Images->image[i][j][k] = 0;
            }
        }
    }
    for(i = 0; i < S1_LENGTH; i++){
        for(j = 0; j < S1_DIMENSIONS; j++){
            for(k = 0; k < S1_DIMENSIONS; k++){
                network->S1_Images->image[i][j][k] = 0;
            }
        }
    }
    for(i = 0; i < C2_LENGTH; i++){
        for(j = 0; j < C2_DIMENSIONS; j++){
            for(k = 0; k < C2_DIMENSIONS; k++){
                network->C2_Images->image[i][j][k] = 0;
            }
        }
    }
    for(i = 0; i < S2_LENGTH; i++){
        for(j = 0; j < S2_DIMENSIONS; j++){
            for(k = 0; k < S2_DIMENSIONS; k++){
                network->S2_Images->image[i][j][k] = 0;
            }
        }
    }
    for(i = 0; i < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; i++){
        network->S2_Vector[i] = 0;
    }
    for(i = 0; i < OUTPUT_LENGTH; i++){
        network->calculated_output[i] = 0;
    }
    for(i = 0; i < OUTPUT_LENGTH; i++){
        network->desired_output[i] = 0;
    }
}

//Sets all parameters within a network to 0
void zero_parameters(cnn* network){
    int i, n, j, k;
    
    for(i = 0; i < C1_LENGTH; i++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                network->C1_Kernels[0][i][j][k] = 0;
            }
        }
    }
    for(i = 0; i < S1_LENGTH; i++){
        for(n = 0; n < C2_LENGTH; n++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    network->C2_Kernels[i][n][j][k] = 0;
                }
            }
        }
    }
    for(j = 0; j < OUTPUT_LENGTH; j++){
        for(k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++){
            network->output_weights[j][k] = 0;
        }
    }
    for(i = 0; i < C1_LENGTH; i++){
        network->C1_Biases[i] = 0;
    }
    for(i = 0; i < C2_LENGTH; i++){
        network->C2_Biases[i] = 0;
    }
    for(i = 0; i < OUTPUT_LENGTH; i++){
        network->output_biases[i] = 0;
    }
    
}

//Generates a uniform random digit within a specified range
double uniform_rand(){
    double value = (double)rand()/RAND_MAX;
    value *= (UPPER - LOWER);
    value += LOWER;
    return value;
}

//Reads header items from input files
void read_headers(FILE *image_ptr, FILE *label_ptr){
    uint32_t label_magic_number = 0u;
    label_magic_number |= getc(label_ptr) << 24;
    label_magic_number |= getc(label_ptr) << 16;
    label_magic_number |= getc(label_ptr) << 8;
    label_magic_number |= getc(label_ptr);

    //Item count from label file
    uint32_t label_item_count = 0u;
    label_item_count |= getc(label_ptr) << 24;
    label_item_count |= getc(label_ptr) << 16;
    label_item_count |= getc(label_ptr) << 8;
    label_item_count |= getc(label_ptr);

    //Magic Number from images file
    uint32_t images_magic_number = 0u;
    images_magic_number |= getc(image_ptr) << 24;
    images_magic_number |= getc(image_ptr) << 16;
    images_magic_number |= getc(image_ptr) << 8;
    images_magic_number |= getc(image_ptr);

    //Item count from images file
    uint32_t images_item_count = 0u;
    images_item_count |= getc(image_ptr) << 24;
    images_item_count |= getc(image_ptr) << 16;
    images_item_count |= getc(image_ptr) << 8;
    images_item_count |= getc(image_ptr);

    //Rows per image
    uint32_t rows = 0u;
    rows |= getc(image_ptr) << 24;
    rows |= getc(image_ptr) << 16;
    rows |= getc(image_ptr) << 8;
    rows |= getc(image_ptr);

    //Columns per image
    uint32_t columns = 0u;
    columns |= getc(image_ptr) << 24;
    columns |= getc(image_ptr) << 16;
    columns |= getc(image_ptr) << 8;
    columns |= getc(image_ptr);
    // printf("Label magic: %u\n",label_magic_number);
    // printf("Label count: %u\n",label_item_count);
    // printf("Image magic: %u\n",images_magic_number);
    // printf("Image count: %u\n",images_item_count);
}

//Reads in an image and it's label from the the input files
void load_image(cnn* network, FILE *image_ptr, FILE *label_ptr){
    int j, k;
    double value = 0;
    for(j = 0; j < INPUT_DIMENSIONS; j++){
        for(k = 0; k < INPUT_DIMENSIONS; k++){
#ifdef USE_RANDOM
            value = fRand();
#else
            value = (double)getc(image_ptr);
            value /= 255;
#endif
            network->input_image[j][k] = value;
        }
    }
#ifdef USE_RANDOM
    double t = (double)(random()%10);
#else
    double t = (double)getc(label_ptr);
#endif
    for(j = 0; j < 10; j++){
        if((int)t == j)
            network->desired_output[j] = 1;
        else
            network->desired_output[j] = 0;
        //printf("%lf\n",network->desired_output[j]);
    }
    //print_image(network->input_image, 28);
}

//Propagate from input to C1
void load_C1(cnn* network){
    //print_image(network->input_image, 28);
    //Create a list of the KERNEL_SIZExKERNEL_SIZE sections of the input image
    int length = C1_DIMENSIONS * C1_DIMENSIONS;
    image_vector* input_sections = new_image_vector(KERNEL_SIZE, length);
    int i, j, k;
    for(i = 0; i < length; i++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                input_sections->image[i][j][k] = network->input_image[(int)floor((double)i/C1_DIMENSIONS) + j][i%C1_DIMENSIONS + k];
            }
        }
        //print_image(input_sections->image[i], 5);
    }
    for(i = 0; i < C1_LENGTH; i++){
        for(j = 0; j < C1_DIMENSIONS; j++){
            for(k = 0; k < C1_DIMENSIONS; k++){
                network->C1_Images->image[i][j][k] = activation(
                        dot_product(input_sections->image[(j*C1_DIMENSIONS)+k], network->C1_Kernels[0][i], KERNEL_SIZE)
                        + network->C1_Biases[i]);
            }
        }
        //print_image(network->C1_Images->image[i], 24);
    }
    free_image_vector(input_sections, KERNEL_SIZE, length);
    free(input_sections);
}

void load_S1(cnn* network){
    int i, n, j, k;
    image_vector* C1_sections = new_image_vector(2, S1_DIMENSIONS*S1_DIMENSIONS);
    //print_image(network->C1_Images->image[0], 24);
    for(i = 0; i < S1_LENGTH; i++){
        for(n = 0; n < 2 * S1_DIMENSIONS*S1_DIMENSIONS; n++){
            if(n%2 == 0){
                for(j = 0; j < 2; j++){
                    for(k = 0; k < 2; k++){
                        C1_sections->image[n/2][j][k] = 
                                network->C1_Images->image[i][(int)floor((double)n/C1_DIMENSIONS)*2 + j][(n%C1_DIMENSIONS) + k];
                    }
                }
                //if(i == 0)
                    //print_image(C1_sections->image[n/2], 2);
            }
        }
        
        for(j = 0; j < S1_DIMENSIONS; j++){
            for(k = 0; k < S1_DIMENSIONS; k++){
                network->S1_Images->image[i][j][k] = average_matrix(C1_sections->image[(j*12) + k], 2);
            }
        }
    }
    //print_image(network->S1_Images->image[0], 12);
    free_image_vector(C1_sections, 2, S1_DIMENSIONS*S1_DIMENSIONS);
    free(C1_sections);
}

void load_C2(cnn* network){
    image_vector* S1_sections[S1_LENGTH];
    int l, m, n, i, j, k;
    int length = C2_DIMENSIONS * C2_DIMENSIONS;
    for(i = 0; i < S1_LENGTH; i++){
        S1_sections[i] = new_image_vector(KERNEL_SIZE, length);
    }
    //print_image(network->S1_Images->image[0], 12);
    for(n = 0; n < S1_LENGTH; n++){
        for(i = 0; i < length; i++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    S1_sections[n]->image[i][j][k] =
                            network->S1_Images->image[n][(int)floor((double)i/C2_DIMENSIONS) + j][i%C2_DIMENSIONS + k];
                }
            }
            //if(n < 2)
                //print_image(S1_sections[n]->image[i], 5);
        }
    }
    
    for(m = 0; m < C2_LENGTH; m++){
        for(n = 0; n < C2_DIMENSIONS; n++){
            for(i = 0; i < C2_DIMENSIONS; i++){
                for(l = 0; l < S1_LENGTH; l++){
                    network->C2_Images->image[m][n][i] +=
                            dot_product(S1_sections[l]->image[(n*C2_DIMENSIONS) + i], network->C2_Kernels[l][m], KERNEL_SIZE);
                }
            }
        }
    }
    
    for(i = 0; i < S1_LENGTH; i++){
        free_image_vector(S1_sections[i], KERNEL_SIZE, length);
    }
    free(*S1_sections);
}

void load_S2(cnn* network){
    int i, n, j, k;
    image_vector* C2_sections = new_image_vector(2, S2_DIMENSIONS * S2_DIMENSIONS);
    //print_image(network->C2_Images->image[0],8);
    for(i = 0; i < C2_LENGTH; i++){
        for(n = 0; n < 2 * S2_DIMENSIONS * S2_DIMENSIONS; n++){
            if(n%2 == 0){
                for(j = 0; j < 2; j++){
                    for(k = 0; k < 2; k++){
                        C2_sections->image[n/2][j][k] = 
                                network->C2_Images->image[i][(int)floor((double)n/C2_DIMENSIONS)*2 + j][n%C2_DIMENSIONS + k];
                    }
                }
                //if(i == 0)
                    //print_image(C2_sections->image[n/2], 2);
            }
        }
        
        for(j = 0; j < S2_DIMENSIONS; j++){
            for(k = 0; k < S2_DIMENSIONS; k++){
                network->S2_Images->image[i][j][k] = average_matrix(C2_sections->image[(j*S2_DIMENSIONS)+k],2);
            }
        }
        //print_image(network->S2_Images->image[i], 4);
    }
    free_image_vector(C2_sections, 2, S2_DIMENSIONS * S2_DIMENSIONS);
}

void concatenate_S2(cnn* network){
    //print_image(network->S2_Images->image[0],4);
    int n, j, k;
    for(n = 0; n < S2_LENGTH; n++){
        for(j = 0; j < S2_DIMENSIONS; j ++){
            for(k = 0; k < S2_DIMENSIONS; k++){
                network->S2_Vector[(n*(S2_DIMENSIONS*S2_DIMENSIONS)) + (j*S2_DIMENSIONS) + k] = network->S2_Images->image[n][j][k];
                //printf("%lf\n",network->S2_Vector[(n*(S2_DIMENSIONS*S2_DIMENSIONS)) + (j*S2_DIMENSIONS) + k]);
            }
        }
    }
}

void load_output(cnn* network){
    int i, n, j, k;
    double value = 0;
    for(i = 0; i < OUTPUT_LENGTH; i++){
        for(n = 0; n < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; n++){
            value += (network->S2_Vector[n] * network->output_weights[i][n]);
        }
        value += network->output_biases[i];
        network->calculated_output[i] = output_activation(value);
        value = 0;
    }
}

void backpropagation(cnn* network, cnn* gradient){
    int i, n, u, v, j, k, l, q;
    double sum = 0;
    //Delta W
    #pragma omp parallel for 
    for(i = 0; i < OUTPUT_LENGTH; i++){
        for(j = 0; j < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; j++){
             gradient->output_weights[i][j] += (network->calculated_output[i] - network->desired_output[i]) * output_activation_derivative(network->calculated_output[i]) * network->S2_Vector[j];
        }
    }
    sum = 0;
    //Delta B
    #pragma omp parallel for 
    for(i = 0; i < OUTPUT_LENGTH; i++){
        gradient->output_biases[i] += (network->calculated_output[i] - network->desired_output[i])
                * output_activation_derivative(network->calculated_output[i]);
    }
    sum = 0;
    //Delta f(j)
    #pragma omp parallel for  
    for(j = 0; j < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; j++){
        for(i = 0; i < OUTPUT_LENGTH; i++){
            gradient->S2_Vector[j] += (gradient->output_biases[i] * network->output_weights[i][j]);
        }
    }
    //un-concatenate
    #pragma omp parallel for 
    for(n = 0; n < S2_LENGTH; n++){
        for(j = 0; j < S2_DIMENSIONS; j++){
            for(k = 0; k < S2_DIMENSIONS; k++){
                gradient->S2_Images->image[n][j][k] = gradient->S2_Vector[(n*S2_DIMENSIONS*S2_DIMENSIONS) + (j*S2_DIMENSIONS) + k];
            }
        }
    }
    //super sample
    #pragma omp parallel for 
    for(n = 0; n < C2_LENGTH; n++){
        for(j = 0; j < C2_DIMENSIONS; j++){
            for(k = 0; k < C2_DIMENSIONS; k++){
                gradient->C2_Images->image[n][j][k] = (0.25) * gradient->S2_Images->image[n][(int)floor((double)j/2)][(int)floor((double)k/2)];
            }
        }
    }
    sum = 0;
    //Delta K2
    #pragma omp parallel for 
    for(n = 0; n < S1_LENGTH; n++){
        for(i = 0; i < C2_LENGTH; i++){
            for(u = 0; u < KERNEL_SIZE; u++){
                for(v = 0; v < KERNEL_SIZE; v++){
                    for(j = 0; j < C2_DIMENSIONS; j++){
                        for(k = 0; k < C2_DIMENSIONS; k++){
                            gradient->C2_Kernels[n][i][u][v] += gradient->C2_Images->image[i][j][k] *
                                    activation_derivative(network->C2_Images->image[i][j][k]) *
                                    network->S1_Images->image[n][j+u][k+v];
                            
                        }
                    }
                }
            }
        }
    }
    //Delta B2
    #pragma omp parallel for
    for(n = 0; n < C2_LENGTH; n++){
        for(j = 0; j < C2_DIMENSIONS; j++){
            for(k = 0; k < C2_DIMENSIONS; k++){
                gradient->C2_Biases[n] +=
                        gradient->C2_Images->image[n][j][k] *
                        activation_derivative(network->C2_Images->image[n][j][k]);
            }
        }
    }
    //Funniest shit i've pulled so consider in debugging
    //Prep for backprop convolution
    image_vector* C2_gradient_temp = new_image_vector(S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    image_vector* C2_network_temp = new_image_vector(S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    #pragma omp parallel for
    for(i = 0; i < C2_LENGTH; i++){
        for(j = 0; j < S1_DIMENSIONS + KERNEL_SIZE - 1; j++){
            for(k = 0; k < S1_DIMENSIONS + KERNEL_SIZE - 1; k++){
                C2_gradient_temp->image[i][j][k] = 0;
                C2_network_temp->image[i][j][k] = 0;
            }
        }
    }
    #pragma omp parallel for
    for(i = 0; i < C2_LENGTH; i++){
        for(j = 4; j < C2_DIMENSIONS + 4; j++){
            for(k = 4; k < C2_DIMENSIONS + 4; k++){
                C2_gradient_temp->image[i][j][k] = gradient->C2_Images->image[i][j-4][k-4];
                C2_network_temp->image[i][j][k] = network->C2_Images->image[i][j-4][k-4];
            }
        }
    }
    //Delta Sp1
    #pragma omp parallel for
    for(n = 0; n < S1_LENGTH; n++){
        for(i = 0; i < S1_DIMENSIONS; i++){
            for(l = 0; l < S1_DIMENSIONS; l++){
                for(q = 0; q < C2_LENGTH; q++){
                    for(u = 0; u < KERNEL_SIZE; u++){
                        for(v = 0; v < KERNEL_SIZE; v++){
                            gradient->S1_Images->image[n][i][l] +=
                                    C2_gradient_temp->image[q][i+u][l+v] *
                                    activation_derivative(C2_network_temp->image[q][i+u][l+v]) *
                                    network->C2_Kernels[n][q][u][v];
                        }
                    }        
                }
            }
        }   
    }
    free_image_vector(C2_gradient_temp, S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    free_image_vector(C2_network_temp, S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    free(C2_gradient_temp);
    free(C2_network_temp);
    
    //super sampling
    #pragma omp parallel for
    for(n = 0; n < C1_LENGTH; n++){
        for(j = 0; j < C1_DIMENSIONS; j++){
            for(k = 0; k < C1_DIMENSIONS; k++){
                gradient->C1_Images->image[n][j][k] = 
                        (0.25) * gradient->S1_Images->image[n][(int)floor((double)j/2)][(int)floor((double)k/2)];
            }
        }
    }
    //Delta K1
    #pragma omp parallel for
    for(n = 0; n < C1_LENGTH; n++){
        for(u = 0; u < KERNEL_SIZE; u++){
            for(v = 0; v < KERNEL_SIZE; v++){
                for(j = 0; j < C1_DIMENSIONS; j++){
                    for(k = 0; k < C1_DIMENSIONS; k++){
                        gradient->C1_Kernels[0][n][u][v] += 
                                gradient->C1_Images->image[n][j][k] *
                                activation_derivative(network->C1_Images->image[n][j][k]) *
                                network->input_image[j+u][k+v];
                    }
                }
            }
        }
    }
    #pragma omp parallel for
    for(n = 0; n < C1_LENGTH; n++){
        for(j = 0; j < C1_DIMENSIONS; j++){
            for(k = 0; k < C1_DIMENSIONS; k++){
                gradient->C1_Biases[n] += gradient->C1_Images->image[n][j][k];
            }
        }
    } 
}

void update_batch_gradient(cnn* image_gradient, cnn* batch_gradient){
    int n, m, j, k;
    for(n = 0; n < C1_LENGTH; n++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                batch_gradient->C1_Kernels[0][n][j][k] +=
                        image_gradient->C1_Kernels[0][n][j][k];
            }
        }
        batch_gradient->C1_Biases[n] += image_gradient->C1_Biases[n];
    }
    for(n = 0; n < S1_LENGTH; n++){
        for(m = 0; m < C2_LENGTH; m++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    batch_gradient->C2_Kernels[n][m][j][k] +=
                            image_gradient->C2_Kernels[n][m][j][k];
                }
            }
            batch_gradient->C2_Biases[m] +=
                    image_gradient->C2_Biases[m];
        }
    }
    for(n = 0; n < OUTPUT_LENGTH; n++){
        for(m = 0; m < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; m++){
            batch_gradient->output_weights[n][m] +=
                    image_gradient->output_weights[n][m];
            
        }
        batch_gradient->output_biases[n] +=
                image_gradient->output_biases[n];
    }
}

void gradient_descent(cnn* network, cnn* gradient){
    int n, m, j, k;
    for(n = 0; n < C1_LENGTH; n++){
        for(j = 0; j < KERNEL_SIZE; j++){
            for(k = 0; k < KERNEL_SIZE; k++){
                network->C1_Kernels[0][n][j][k] -=
                        LEARNING_RATE*(gradient->C1_Kernels[0][n][j][k]/BATCH_SIZE);
            }
        }
        network->C1_Biases[n] -= LEARNING_RATE*(gradient->C1_Biases[n]/BATCH_SIZE);
    }
    for(n = 0; n < S1_LENGTH; n++){
        for(m = 0; m < C2_LENGTH; m++){
            for(j = 0; j < KERNEL_SIZE; j++){
                for(k = 0; k < KERNEL_SIZE; k++){
                    network->C2_Kernels[n][m][j][k] -=
                            LEARNING_RATE*(gradient->C2_Kernels[n][m][j][k]/BATCH_SIZE);
                }
            }
            network->C2_Biases[m] -=
                    LEARNING_RATE*(gradient->C2_Biases[m]/BATCH_SIZE);
        }
    }
    for(n = 0; n < OUTPUT_LENGTH; n++){
        for(m = 0; m < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; m++){
            network->output_weights[n][m] -=
                    LEARNING_RATE*(gradient->output_weights[n][m]/BATCH_SIZE);
            
        }
        network->output_biases[n] -=
                LEARNING_RATE*(gradient->output_biases[n]/BATCH_SIZE);
    }
}

void free_image_vector(image_vector* images, int dimensions, int length){
    int i;
    for(i = 0; i < length; i++){
        free_array(images->image[i], dimensions);
        //free(images->image[i]); //chath removed double free
    }
    free(images->image);
}

void free_array(array array, int rows){
    int j, k;
    for(j = 0; j < rows; j++){
        free(array[j]);
    }
    free(array);
}

//Computes the average value of a matrix
double average_matrix(array image, int dimensions){
    int j, k;
    double value = 0;
    for(j = 0; j < dimensions; j++){
        for(k = 0; k < dimensions; k++){
            value += image[j][k];
        }
    }
    value /= (dimensions * dimensions);
    return value;
}

void print_image(array image, int dimensions){
    int j, k;
    for(j = 0; j < dimensions; j++){
        for(k = 0; k < dimensions; k++){
            if(image[j][k] > 0)
                printf(RED"[%.2lf]",image[j][k]);
            else
                printf(RESET"[%.2lf]",image[j][k]);
        }
        printf(RESET"\n");
    }
    printf(RESET"\n");
}

//Computes the dot product of two identically sized matrices
double dot_product(array array1, array array2, int dimensions){
    int j, k;
    double value = 0;
    for(j = 0; j < dimensions; j++){
        for(k = 0; k < dimensions; k++){
            value += array1[j][k] * array2[j][k];
        }
    }
    return value;
}

double activation(double x){
    if(x < 0)
        return 0.01*x;
    return x;
}

double activation_derivative(double x){
    if(x < 0)
        return 0.01;
    return 1;
}

double output_activation(double x){
    return (double)1/(1+exp(-x));
}

double output_activation_derivative(double x){
    return x * (1 - x);
}

double loss_function(cnn* network){
    int i;
    double value = 0;
    for(i = 0; i < 10; i++){
        value += (network->calculated_output[i] - network->desired_output[i])
                * (network->calculated_output[i] - network->desired_output[i]);
        //printf("%lf\n",network->calculated_output[i]);
                
    }
    return value/2;
}
