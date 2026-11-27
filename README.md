# Group Project: Designing and Optimizing a Deep Neural Network

## Background

- The temperature; pH; glucose, glutamine, lactate, and ammonia concentrations; and viable cell density are among the
  most important factors influencing cell culture dynamics.


- The growth rate and death rate of cells, and the production rate of recombinant proteins are particularly affected by
  these factors.

## Task

- Your main task is to design a fully connected neural network and optimize its hyperparameters to:
    - Predict:
        - Specific Growth Rate [d^-1]
        - Specific Death Rate [d^-1]
        - Specific Production Rate [pg cell^-1 d^-1]
    - Based on:
        - Temperature [C]
        - pH
        - Glucose [mM]
        - Glutamine [mM]
        - Lactate [mM]
        - Ammonia [mM]
        - Viable Cell Density [cells/mL]


- In your GitHub repository, add a description in the "About" section. (10 - 20 words)


- You must also include a `README.md` file which has the following structure:
### Predicting Cell Life Cycles using an Optimized Deep Neural Network
This project develops and evaluates a fully connected deep neural network to predict key bioprocess performance indicaotrs using experimental cell growth data. The model uses seven input variables: temperature, pH, glucose, glutamine, lactate, ammonia, and viable cell density (VCD) to predict specific growth rate, specific death rate, and specific production rate. Hyperparameter optimization and cross-validation are used to identify an effective network architecture and evaluate predictive performance. The optimized neural network is compared against polynomial regression to assess prediction accuracy and computational efficiency. The overall objective is to investigate the application of machine learning to bioprocess modelling and optimization.
      
## Highlights
- Developed a fully connected neural network to predict specific cell growth, death, and production rates from seven bioprocess variables. 
- Optimized five neural network hyperparameters using a grid search of 48 configurations. 
- Evaluated model performance using four-fold cross-validation and NRMSE. 
- Investigated the influence of individual hyperparameters on prediction accuracy using three-fold cross-validation. 
- Compared the optimized neural network against polynomial regression to assess predictive accuracy and computational efficiency.

## Environment 
- Python: 3.14.7
- NumPy: 2.5.2
- Pandas: 3.0.5 
- Matplotlib: 3.11.0 
- Scickit-learn: 1.9.0
- Torch: 2.13.0


    
    - **Code Design**
        - Heading 2
        - 500 - 1500 words
        - Describe the workflow of the program after `main.py` is executed.
        - Explain the purpose of each major stage of the workflow and why it exists.
        - Examples of major stages include (but are not limited to) data loading, preprocessing, feature engineering,
          model training, etc.
        - If `main.py` executes code located elsewhere in your repository (e.g., `src/classes.py`), then that code
          should also be described.
        - Organize this section using Heading 3 subsections.
        - For each subsection, describe:
            - What does this stage of the workflow do?
            - Why is this stage necessary?
            - How does this stage contribute to the overall workflow?
            -
## Analysis
The optimal model was found to have the following hyperparameters:
- the hidden layers architecture had 6 hidden layers of the form (8, 16, 32, 16, 8, 4)
- the activation function was the hyperbolic tangent function
- the learning rate was 0.001
- the batch size was 64
- the weight decay was 0.0

In determining the optimal values, a train-test split was utilized, rather than cross-validation. 400 epochs were used for training; [justify]. 
Within the search space, param_grid, several combinations were tested. 
- The hidden layers architecture initially searched (5,), (6, 5, 4), (8, 16, 32, 16, 8, 4). The number of neurons in each layer was kept relatively similar to the number of inputs and outputs. The total number of inputs and outputs was 10 (7 features/inputs and 3 targets/outputs). The tuple (5,) took the mean of the inputs and outputs and provided a small and simple starting point; (6, 5, 4) decreased the number of neurons withen the hidden layers; (8, 16, 32, 16, 8, 4) was a deeper network that both increases and decreases the number of neurons.
- The tanh and ReLU functions were initially tested. The tanh function is generally easier to train models when compared to the sigmoid function, but has a vanishing gradient (i.e., when large positive or negative values are input, the gradient vanishes and hinders the process of training the model). This is because the tanh function produces zero-centred output values at (-1, 1). The ReLU function does not have a vanishing gradient with postive inputs and is computationally efficient. By using both functions, we can compare the advantages and limitations of each, to help support the final choice within our neural network.
- The learning rates used in this section were 0.0001, 0.001, and 0.01. When learning rates are too small, many updates are required before the minimum point is reached. Conversely, when the learning rate is too high, drastic updates are undergone, which can lead to divergent behaviour. The optimal learning rate will swiftly reach the minimum point. The values used provided a good starting point, exploring smaller, larger, and intermediate values.
- The explored batch sizes were 64 and 256. When choosing a batch size, consideration must be made to the computational efficiency, gradient noise, convergence speed, and generalization. Smaller batch sizes can have higher levels of noise which can help the optimizer escape poor local minima and improve generalization. Larger batch sizes are more accurate in terms of gradient estimates, but can converge to solutions that generalize less well and often require different learning rates. Therefore, both a smaller and larger batch size were chosen.
- The weight decays that were searched were 0 and 0.01. These values explore zero decay and non-zero decay to investigate their impacts on the weights of the model. Weight decay reduces the risk of having extreme weights, in turn reducing the risk of overfitting.

In total, 72 combinations of hyperparameters were tested. 3 hidden layer architectures, 2 activation functions, 3 learning rates, 2 batch sizes, and 2 weight decays were tested. This distribution allowed for a fairly representative search, with computational efficiency still in mind.

The final set of optimal hyperparameters was chosen as it provided the lowest normalized root-mean-square error (NRMSE) across all three output variables. The standardization of the NRMSE allows for normalization of the targets to the interquartile range of the test data, in order to allow for a representative comparison across the different output scales and magnitudes.  
 
### Cross-Validation
- A 4-fold cross-validation was performed on the optimal network, using the hyperparameter values from the previous section. Cross-validation allows for the robust evaluation of the model's performance across multiple partitions, reducing the evaluation metric's (i.e., RMSE and NRMSE) dependence on any single split. The means and standard deviations of the RMSE and NRMSE for each target variable are calculated. A summary of the results are shown in Table 1.
 
#### Table 1: Summary of Means and Standard Deviations for Specific Growth, Death, and Production Rates

|Target                                    |RMSE                 |RMSE                  |NRMSE              |NRMSE               |
|------------------------------------------|---------------------|----------------------|-------------------|--------------------|
|mean                                      |std                  |mean                  |std                |                    |
|Specific Death Rate [d^-1]                |0.0032261134974553023|2.7360058892960186e-05|0.28710687765901016|0.010320278977173494|
|Specific Growth Rate [d^-1]               |0.026667057638327378 |0.0002478370462714555 |0.40134703101760183|0.014838824056510498|
|Specific Production Rate [pg cell^-1 d^-1]|2.453143702074357    |0.004165929420499475  |0.46560336252683243|0.01373742637076793 |

- To ensure the number of epochs was sufficient, we can visualize the 4-fold cross validation.

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/c7861cee-8ec7-43da-af13-ef6060c03cfc" />

#### Figure 1: Training and validation loss over 400 epochs for each of the four folds during 4-fold cross-validation of the optimized neural network.

- As seen, the loss converges after approximately 100 epochs for each cross-validation fold. Thus, we can conclude that the model had sufficient time to converge across the 400 epoch specification, and increasing this number any further would be unlikely to provide any significant benefit. 

### Influence of Hyperparameters 
To understand the impact of each individual hyperparameter, the optimized neural network is used as the baseline; each hyperparameter is changed individually while all other values are held constant. 3-fold cross-validation is used to reduce computation time. The means and standard deviations for the RMSE and NRMSE are computed for each hyperparameter. 
**Hidden Layers**
To examine the influence of the hidden layers on model error, the optimal architecture alongside 2 less complex and 2 more complex architectures are tested. The results are shown in Table 2. 

#### Table 2: Summary of Means and Standard Deviations of RMSE and NRMSE for Hidden Layer Architectures of Varying Complexity

|Parameter    |Value                  |Mean RMSE         |Std RMSE            |Mean NRMSE         |Std NRMSE           |
|-------------|-----------------------|------------------|--------------------|-------------------|--------------------|
|hidden_layers|(4, 8, 16, 8, 4, 2)    |1.0788685781493896|0.16702347797306313 |0.49581004712358917|0.08002573661021435 |
|hidden_layers|(6, 12, 24, 12, 6, 3)  |0.8384787311974974|0.018717777710998956|0.39243463169325404|0.01671818277828097 |
|hidden_layers|(8, 16, 32, 16, 8, 4)  |0.8301625034321937|0.008692520685285648|0.3848911755759455 |0.0123013310037532  |
|hidden_layers|(10, 20, 40, 20, 10, 5)|0.8330192156804808|0.006255700684871613|0.38518902976727176|0.012447381926708759|
|hidden_layers|(12, 24, 48, 24, 12, 6)|0.842320207365558 |0.006335436233308696|0.38954995679739896|0.013312848254386347|

As seen, the mean RMSE and NRMSE as well as the standard deviation of the NRMSE reach their minimum at the optimial hidden layers architecture. The standard deviation of the RMSE is at a minimum at the 4th most complex hidden layer architecture. Thus, we can confirm that the optimal chosen architecture is complex enough to generally provide the lowest error in terms of our evaluation metrics. Increasing the complexity beyond this does not prove to be significantly beneficial. 

**Activation**
In inspecting the effects of the activation function, four different activation functions are compared to one another: the sigmoid function, the hyperbolic tangent (tanh) function, the rectified linear unit (ReLU) function, and the leaky ReLU function. Summary statistics for each function are shown in the following table. 

#### Table 3: Summary of Means and Standard Deviations of RMSE and NRMSE for the Sigmoid, Tanh, ReLU, and LeakyReLU Activation Functions

|Parameter    |Value                  |Mean RMSE         |Std RMSE            |Mean NRMSE         |Std NRMSE           |
|-------------|-----------------------|------------------|--------------------|-------------------|--------------------|
|activation   |sigmoid                |0.9149184423522604|0.01007964203971543 |0.46489324051221614|0.01313671115743961 |
|activation   |tanh                   |0.8301625034321937|0.008692520685285648|0.3848911755759455 |0.0123013310037532  |
|activation   |ReLU                   |0.8537698166964272|0.011520569170311502|0.3950952709924483 |0.011245411086501178|
|activation   |leakyReLU              |0.8419953616802814|0.008538979377771127|0.39151198748449895|0.014054231357472783|

The tanh function provides the minimum mean RMSE and NRMSE, demonstrating that it provides the best predictions between the tested activation functions. The LeakyReLU function has the lowest standard deviation of the RMSE whereas the ReLU function has the lowest standard deviation of the NRMSE. Nevertheless, the results are comparable between the tanh, ReLU, and LeakyReLU activation functions. The tanh function is generally more computationally expensive than the ReLU and LeakyReLU functions. However, the improved prediction accuracy justified the choice of the tanh function as the model's optimal choice. 

**Learning Rate**

#### Table 4: Summary of Means and Standard Deviations of RMSE and NRMSE for Varied Learning Rates

|Parameter    |Value                  |Mean RMSE         |Std RMSE            |Mean NRMSE         |Std NRMSE           |
|-------------|-----------------------|------------------|--------------------|-------------------|--------------------|
|learning_rate|0.0001                 |0.8718008501626541|0.02420687146553957 |0.40828234878332187|0.01672067707532705 |
|learning_rate|0.0005                 |0.8329242399098211|0.014486950932482654|0.3864621314538766 |0.01401109586475799 |
|learning_rate|0.001                  |0.8301625034321937|0.008692520685285648|0.3848911755759455 |0.0123013310037532  |
|learning_rate|0.002                  |0.837024724688363 |0.009175022073819391|0.38809223133710646|0.012834237507298068|
|learning_rate|0.01                   |0.8671476939264675|0.012683827688719633|0.40349131604418015|0.01720719062887714 |

The optimized learning rate of 0.001 consistently produced the lowest mean and standard deviation RMSEs and NRMSEs. Both lower and higher learning rates provided increased error rates. Furthermore, to avoid overfitting and underfitting, the choice of 0.001 remains the best choice for the network. 

**Batch Size**

#### Table 5: Summary of Means and Standard Deviations of RMSE and NRMSE for Increasing Batch Sizes

|Parameter    |Value                  |Mean RMSE         |Std RMSE            |Mean NRMSE         |Std NRMSE           |
|-------------|-----------------------|------------------|--------------------|-------------------|--------------------|
|batch_size   |16                     |0.837258654529848 |0.013135484130335372|0.38889055256095467|0.015106154801460594|
|batch_size   |32                     |0.8358007290867965|0.008140070538719183|0.38612703759036715|0.012784333163493233|
|batch_size   |64                     |0.8301625034321937|0.008692520685285648|0.3848911755759455 |0.0123013310037532  |
|batch_size   |128                    |0.8357417826304802|0.012944560662082225|0.3868721172769767 |0.012668644767355823|
|batch_size   |256                    |0.8376604253808949|0.012938589651861163|0.38868357552054306|0.013743642352504226|

The optimal batch size of 64 provides the lowest mean RMSE and NRMSE and standard deviation NRMSE. The error across all evaluation metrics for each tested batch size showed small differences, suggesting that batch size has a minimal impact on the accuracy of the neural network. 

**Weight Decay**

#### Table 6: Summary of Means and Standard Deviations of RMSE and NRMSE for Increasing Weight Decay Values

|Parameter    |Value                  |Mean RMSE         |Std RMSE            |Mean NRMSE         |Std NRMSE           |
|-------------|-----------------------|------------------|--------------------|-------------------|--------------------|
|weight_decay |0                      |0.8301625034321937|0.008692520685285648|0.3848911755759455 |0.0123013310037532  |
|weight_decay |0.0001                 |0.8301704940067384|0.00880525048184499 |0.3848954650759864 |0.012323244664212696|
|weight_decay |0.001                  |0.8303791684257392|0.009538388056742102|0.38495198764751687|0.012452281723966188|
|weight_decay |0.01                   |0.832107907358337 |0.012199385666528548|0.3854089961460702 |0.012990782339559104|
|weight_decay |0.1                    |0.8800568182202984|0.02114737786213205 |0.4071090987351163 |0.013655672369881225|

Using a weight factor of 0 returned the lowest error across all evaluation metrics. The error is relatively consistent across all weights, again suggesting a minimal impact on the model's performance; however, error increases relatively significantly at a weight of 0.1. This indicates that when the model's weights are shrunk towards zero, the accuracy of the model is hindered, and potentially leading to overfitting. 

### Theoretical Improvements 
More combinations of hyperparameters, more epochs, 
### Comparable Model Performance
Our optimized model can be compared to an optimized polynomial regression model to 


      
        - Based on all the data you have collected on the influence of these hyperparameters on model error, discuss
          what next steps you would take to improve the model if you had access to more computational power.
        - Compare the performance of your optimized neural network to that of an optimized polynomial regression model.
            - You do not need to do anything fancy here, you can recycle old code from the course or past assignments.
            - Discuss the difference in model performance and training time. You do not need to provide exact values for
              training time, just describe the order of magnitude (e.g., seconds vs minutes vs hours, etc.).
- Hints:
    - A big challenge with this project is managing your computational resources effectively.
    - Start with a small set of hyperparameters and a lower number of cross-validation folds to test your code and see
      that everything is working properly. You may even start by using a subset of the dataset.
    - Another approach to speeding up code development can be to temporarily lower the number of epochs to something
      like 10. However, this will greatly diminish the performance of your model, so only do this to test if your code
      works without error, then increase the number of epochs to a more reasonable value.
    - Once you see that your code runs without errors, use the entire dataset, add more hyperparameters, and use the
      appropriate number of cross-validation folds for each section.
    - Save results into tables so that you don't need to run the entire code again each time.
    - For example, if you want to make a figure, you don't want to have to run the entire neural network hyperparameter
      optimization each time. Store the values that you want to plot in a table. Then plot from the table each time you
      make changes to your figures.
    - Once your entire workflow is complete you may run your entire code again one last time to ensure that everything
      is consistent.


- Reminders:
    - Only your `main.py` script will be run to test your code in the course environment.
    - You are free to use the libraries found in the course environment and code from the course repository as needed.
    - You are primarily being evaluated on your ability to justify your decisions rather than arriving at the absolute
      optimal solution.
    - Word limits should be viewed as strong suggestions, rather than hard limits.
    - Figures should be embedded in the README. They should not be a link to a file elsewhere in the repository.
    - All figures should be in the `figures` directory.
    - You should present your repository like a portfolio project rather than an assignment for a course. I would
      encourage you not to make any references to the course (e.g., course code, course name, assignment number, student
      ID, etc.).
    - You can find a guide to the major Markdown syntax elements here: https://www.markdownguide.org/cheat-sheet/
    - You can use this website to convert CSV to Markdown, if needed: https://convertcsv.com/csv-to-markdown.htm

## Setting up your assignment

1. Download the [repository template](https://github.com/Shawn-Chahal/chg4360c-repo-template) for the course.


2. Add [`dataset_cell_growth_7factor.csv`](provided_files/dataset_cell_growth_7factor.csv) to the `datasets` directory
   in your local repository.


3. When you open the project in PyCharm, make sure that you have the course environment selected in the bottom-right
   corner.


4. Make sure to replace any placeholder files from the repository template with the appropriate content.


5. Delete any placeholder files that you are not using.


6. Push the project to a new GitHub repository. Keep the repositroy name short (15 - 30 characters or 2 - 4 words).

## Provided files

[`dataset_cell_growth_7factor.csv`](provided_files/dataset_cell_growth_7factor.csv)

- This file contains simulated bioprocess data with the following columns:
    - `Specific Growth Rate [d^-1]`: The specific growth rate of cells in units of d^-1.
    - `Specific Death Rate [d^-1]`: The specific death rate of cells in units of d^-1.
    - `Specific Production Rate [pg cell^-1 d^-1]`: The specific production rate of recombinant protein in units of pg
      cell^-1 d^-1.
    - `Temperature [C]`: The temperature in units of °C.
    - `pH`: The pH.
    - `Glucose [mM]`: Glucose concentration in units of mM.
    - `Glutamine [mM]`: Glutamine concentration in units of mM.
    - `Lactate [mM]`: Lactate concentration in units of mM.
    - `Ammonia [mM]`: Ammonia concentration in units of mM.
    - `Viable Cell Density [cells/mL]`: The concentration of living cells in units of cells/mL.

## Submission Requirements

- Submit a link to your GitHub repository by email:
    - To: TA (see Brightspace)
    - CC: All group members
    - Subject: CHG 4360C Group Project Submission


- The timestamp of your submission will be whichever is latest between the timestamp of the last commit in your GitHub
  repository and the timestamp of your Brightspace submission.


- The GitHub repository only needs to be uploaded to one group member's GitHub profile, however each group member is
  encouraged to copy the repository to their respective GitHub profiles for their own portfolio and reference.

## Rubric

| Component         | Weight |
|-------------------|--------|
| Code Execution    | 25%    |
| GitHub Repository | 10%    |
| Introduction      | 5%     |
| Code Design       | 20%    |
| Analysis          | 40%    |

Introduction consists of: Project Name, Highlights, and Environment.
