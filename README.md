**INTRO**

This project is a test between classical and quantum ML in the application of environmental modeling of soil health values to pre-flash flood modeling. The dataset used was from the open source US Department of the Interior Data (for more info, access the following link [https://www.sciencebase.gov/catalog/item/56608ee4e4b071e7ea544e04.](url))

Quantum physics and its dependent engineering systems has interested me and applying it to modeling where randomness is a problem to accuracy could be utilized by the superposition of randomness created by quantum ML.

This is where I decided to test regression of data of different soil nutrient values from necessary elements like NO3, P, K, and others along with toxic metals like Al, Cd, and Mn to days before a flash flood.


**RESULTS**

ANN (classical) performed better than QNN (quantum) due to preset regression like exponential, polynomial, and linear along with a shorter period of time. However, QNNs did perform better in R^2 (accuracy) in some cases although still taking more time. Specifically, when layer length of qubits were increased, QNN runtimes increased by more than double than without the quantum circuits. Along so, sometimes data was underfitted and wasn't shown the right way. This can be seen in the regression Python notebooks.

_Quantum Regression (Sourced from this project)_
<img width="708" height="554" alt="Screenshot 2025-07-11 at 10 41 52 PM" src="https://github.com/user-attachments/assets/b3479cee-4f46-4413-a96b-ad13fbfba50a" />

_Classical Regression (Sourced from this project)_
<img width="1402" height="460" alt="Screenshot 2025-07-11 at 10 43 21 PM" src="https://github.com/user-attachments/assets/1e406747-2add-40d0-97b7-f92d8247f3d3" />


Attempting to run the quantum neural network in whole using the full python file also didn't yield good results, indicating a still developing field of quantum computing and modeling in the field of neural networks.
