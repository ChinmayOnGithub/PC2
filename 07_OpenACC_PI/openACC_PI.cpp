#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <math.h>
#include <ctime>

#define N 1000000

int main()
{
    int count = 0;

    // Start timing using clock()
    clock_t start_time = clock();

#pragma acc parallel loop reduction(+ : count)
    for (int i = 0; i < N; i++)
    {
        double x = ((double)rand() / RAND_MAX);
        double y = ((double)rand() / RAND_MAX);
        double inCircle = x * x + y * y;

        if (inCircle <= 1.0)
        {
            count++;
        }
    }

    double pi = 4.0 * count / N;

    // End timing using clock()
    clock_t end_time = clock();

    double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("Time taken: %f ms\n", time_taken); // Output in milliseconds
    printf("Estimated value of Pi: %f\n", pi);
    printf("Accuracy: %f%% error from actual Pi\n", fabs(pi - M_PI) / M_PI * 100);

    /*
    // Uncomment to show more details about the calculation process
    printf("\n--- Monte Carlo PI Calculation Details ---\n");
    printf("Total points generated: %d\n", N);
    printf("Points inside circle: %d\n", count);
    printf("Points outside circle: %d\n", N - count);
    printf("Ratio (inside/total): %f\n", (double)count / N);
    printf("PI estimate (4 * ratio): %f\n", pi);
    printf("Actual PI value: %f\n", M_PI);
    printf("Absolute error: %f\n", fabs(pi - M_PI));
    printf("Relative error: %f%%\n", fabs(pi - M_PI) / M_PI * 100);

    // Sample calculation verification
    printf("\nVerification of first 10 random points:\n");
    srand(time(NULL)); // Reset random seed for repeatability of this section
    for (int i = 0; i < 10; i++) {
        double x = ((double) rand() / RAND_MAX);
        double y = ((double) rand() / RAND_MAX);
        double distance = x * x + y * y;
        printf("Point %d: (%f, %f) - Distance from origin squared: %f - %s\n",
               i+1, x, y, distance, (distance <= 1.0) ? "Inside" : "Outside");
    }
    */

    return 0;
}
