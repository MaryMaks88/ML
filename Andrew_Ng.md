<a name="top"></a>

Machine learning algorithms:

- Supervised learning - giving your model examples,
                        a relevant pair of input (X) and output (y),
                        when your model knows a correct answer,
                        and learns according to the given answers.

Regression - to predict a number (house pricing, temperature...)

Options of choosing which function to take to predict house price.
Linear or curve.

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8eedc21b-09da-418a-919d-968ef04229e5" /></p>

Classification - to predict a class (spam/not, yes/no...), 
or multiclass classification -> many different classes. 

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5c2a8442-5059-4fc6-a874-9a7588da6d12" /></p>

- Unsupervised learning - not giving your model expected output examples.
                          Model should find the group patterns by itself,
                          and spare the data into categories (clusters).

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/40ba5a21-5cb9-4837-81e5-cd1782158ba8" /></p>

How clusterization may look like

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/2dd08fc0-c663-4eca-8bbf-9f56b1295292" /></p>


Linear Regression
Builds a model which establishes a relationship between features and targets (as a line).
For simple linear regression, the model has two parameters 𝑤 and 𝑏,
whose values are 'fit' using training data.

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/0ddb3e5d-1add-48b1-978f-b909c21ac5f1" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/cd8485f6-59aa-4285-9a17-422e189ae882" /></p>

Cost function - shows how well the model is doing, so we could try to do it better.

Weights (slope of the line), biases

Cost is a measure of how well our model is predicting the target price of the house. The term 'price' is used for housing data.

The equation for cost with one variable is:

<p><img width="400" height="150" alt="image" src="https://github.com/user-attachments/assets/3b3ef150-d61c-4751-8a5c-513d30864b54" /></p>


where
𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏(2)

𝑓𝑤,𝑏(𝑥(𝑖))
  is our prediction for example  𝑖
  using parameters  𝑤,𝑏
 .
(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2
  is the squared difference between the target value and the prediction.
These differences are summed over all the  𝑚
  examples and divided by 2m to produce the cost,  𝐽(𝑤,𝑏)

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/c4242bc1-ad35-4d73-bc81-90f94cadc593" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/99591771-ce34-4a5b-91b5-8e1f4df15de0" /></p>

Cost function -> example

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/4cfb9f1a-3dfb-4b93-b330-ba9441d7c5a0" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8d0f3d36-260c-4487-acc8-04503e360ffb" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/59ee1676-f21e-4b30-9650-823c00b4de50" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/eda7e919-0ff8-462e-902e-d755ec24ce3d" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/61a4d54a-f9f5-41ec-a31e-c694c9d18c62" /></p>

Gradient Descent - keep changingw, b to reduce J(w, b),
                  until we settle at or near a minimum

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/06ba6c0c-020e-4df2-8e87-e3fc8ab55300" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8b929cf8-bbc3-4ad5-970c-6a3ebb596ac0" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/3f27d76e-38a4-4a97-b6a3-ce1b181ab954" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/a25277c0-ae2a-4084-b127-66ae70245731" /></p>

Learning rate

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/70786741-50a6-46af-92c1-cf7077539206" /></p>

With a few local minimums GD will reach one, at that moment weights will stop to change.
So GD will keep the last w value he got.

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/859ca6bb-4f66-4875-8245-d16e8094e202" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/e61c7792-d4c1-4ce7-8b1e-e7fb578ed564" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/31baef43-d0ba-4f40-b7fe-0f9a06501f2a" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/c3dd9a9e-fea3-426f-bf89-d28f99d59583" /></p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/214b9a76-dbdc-44b6-886e-73302fce6dc0" /></p>

















[Нагору ↑](#top)
