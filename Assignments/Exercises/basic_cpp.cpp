#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

void print_hello_world(){
    std::cout << "Hello World\n";
}

void print_even(int N){
    for(int number=0; number<=N; number+=2){
        std::cout << number;
    }
}

void print_odd(int N){
    for(int number=1; number<=N; number+=2){
        std::cout << number;
    }
}

float sphere_volume(float radius){
    float pi = 3.14;
    float volume = 4 / 3 * pi * radius * radius * radius;
    return volume;
}

std::tuple<int, int> quotient_and_remainder(int numerator, int denominator){
    return std::make_tuple(numerator / denominator, numerator % denominator);
}

float sum_vector_elements(std::vector<int> &vec){
    int sum = 0;
    for(int &v: vec){
        sum += v;
    }
    return sum;
}

void square_root_vector_elements(std::vector<float> &vec){
    for(float &v: vec){
        v = std::pow(v, 0.5);
    }
}

void swap_values(float &a, float &b){
    float temp_val;
    temp_val = a;
    a = b;
    b = temp_val;
}

void sort_vector_elements(std::vector<int> &vec){
    std::sort(vec.begin(), vec.end());
}

// Ex 11
class Rectangle{
public:
    float width;
    float height;
    float area(float width, float height);
};

// Ex 12
float Rectangle::area(float width, float height)
{
    return width * height;
}

// Ex 13
int matrix_sum(std::vector<double> two_matrices){
    // Probably does not work!
    int first_matrix = two_matrices[0];
    int second_matrix = two_matrices[1];
    return first_matrix + second_matrix;
}

void add_two_matrices(double matrix1[3][3], double matrix2[3][3], double resulting_matrix[3][3]){
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            resulting_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}


int main(){
    // Ex 1
    std::cout << "Ex 1\n";
    print_hello_world();  

    // Ex 2
    std::cout << "\nEx 2\n";
    print_even(20); 

    // Ex 3
    std::cout << "\n\nEx 3\n";
    print_odd(20);

    // Ex 4
    std::cout << "\n\nEx 4\n";
    float volume = sphere_volume(1);
    std::cout << "Volume for r=1 is: " << volume << "\n";

    // Ex 5
    std::cout << "\nEx 5\n";
    int quotient, remainder;
    std::tie(quotient, remainder) = quotient_and_remainder(10, 3);
    std::cout << "Quotient, remainder for 10/3: " << quotient << remainder << "\n";

    // Ex 6
    std::cout << "\nEx 6\n";
    std::vector<int> test_vector = std::vector(5, 2);
    float vector_sum = sum_vector_elements(test_vector);
    std::cout << "Vector elements: ";
    for (int &v: test_vector){
        std::cout << v;
    }
    std::cout << ", Element sum: " << vector_sum << "\n";

    // Ex 7
    std::cout << "\nEx 7\n";
    std::vector<float> test_vector_2 = {2.0f, 3.0f, 4.0f, 5.0f};
    square_root_vector_elements(test_vector_2);
    std::cout << "Square root of vector elements";
    for (auto &sqrt_v: test_vector){
        std::cout << sqrt_v;
    }

    // Ex 8
    std::cout << "\n\nEx 8\n";
    // OBS OBS OBS

    // Ex 9
    std::cout << "\nEx 9\n";
    float sample_float_1 = 1.0f;
    float sample_float_2 = 2.0f;
    std::cout << sample_float_1 << "" << sample_float_2;
    swap_values(sample_float_1, sample_float_2);
    std::cout << ", Swapped: " << sample_float_1 << "" << sample_float_2 << "\n";

    // Ex 10
    std::cout << "\nEx 10\n";
    std::vector<int> test_vector_two = {1, 2, 5, 4, 3};
    for (int n: test_vector_two){
        std::cout << n << " ";
    }
    std::cout << "\n";
    sort_vector_elements(test_vector_two);
    for (int n: test_vector_two){
        std::cout << n << " ";
    }
    std::cout << "\n";

    // Ex 11
    std::cout << "\nEx 11\n";
    std::cout << "*Class implemented*\n";

    // Ex 12
    std::cout << "\nEx 12\n";
    float rectangle_width = 2.0f;
    float rectangle_height = 3.0f;
    Rectangle MyRectangle;
    MyRectangle.width = rectangle_width;
    MyRectangle.height = rectangle_height;
    float rectangle_area = MyRectangle.area(rectangle_width, rectangle_height);
    std::cout << "Rectangle area: " << rectangle_area << "\n"; 

    // Ex 13
    std::cout << "\nEx 13\n";
    double matrix1[3][3] = {{1, 2, 3}, {3.3, 3.3, 3.3}, {-1, -1, -1}};
    double matrix2[3][3] = {{3, 2, 1}, {-3.3, -3.3, 0}, {1, 2, 3.0}};
    double matrix_sum[3][3];
    add_two_matrices(matrix1, matrix2, matrix_sum);
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            std::cout << i << j << ": " << matrix1[i][j] << "+" << matrix2[i][j] << "=" << matrix_sum[i][j] << "\n";
        }
    }
    return 0;
}


