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

Multiple Features (variables)

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/9ba39cd6-39a7-4e8c-a5e8-4a198552bc3f" />
</p>

Below schema shows us that each weight for each feauter represents how this feature impacts the target. 
That each size extension will push the price up on 0.1 (100$),
each bedroom will increase price on 4 (4000$), each new floor will increase price on 10 (10000$), 
and each year will decrease the price on -2 (-2000$).
Bias shows us the minimum price we could pay for the object where all weights are 0,
we have no size, no bedrooms etc.

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/124080e6-56e4-40f2-9d9e-3763c2631b7d" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/031e19db-3936-4ef0-a7f4-7d5842f776d4" />
</p>

Vectorization in code

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/1b4bb674-8268-4f47-bdfc-55ce206299f2" />
</p>

Parallel calculation 

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/0c23713f-6eb8-4ea9-a184-7f98efa0ee05" />
</p>

Vectorization in Gradient Descent

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/0c4ad311-ced5-4831-898d-7246cb4f0a84" />
</p>

Gradient Descent for MUltiple Regression

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/bb151f12-13b7-4576-8b5a-b07cc01bebdf" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/ddffdc14-1929-4174-a2d2-9e8eb3f9ac06" />
</p>

Feature Scaling

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/c2b99497-7e90-4ddc-80b6-fa62f33e4933" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/ceca0807-820e-4b4e-bf2a-324bbd080f90" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/c21d398a-3f9d-4d5b-b5b3-81525622d0b6" />
</p>

Mean normalization

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5d77a411-bbe7-477e-923e-4cb85c086c74" />
</p>

Z - score normalization (Standardization)

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/16680ae6-820f-468a-af43-575152e7335b" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/7200aa9f-21c2-411e-b23c-3e170a2ed5a7" />
</p>

Learning curve

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/fe4c67ae-2a4a-4ada-8e65-281c1800cdf3" />
</p>

Choosing the learning rate

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/6b1c192c-0c4f-419e-be93-bbc66c29cf31" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/e0c0832d-cd57-4db4-b088-016d43e4fb2b" />
</p>

Feature Engineering

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/3474cf57-70d7-46b3-970b-aed9cfadb5f4" />
</p>

Polynomial Regression

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/026245a1-4189-461c-8f45-438becd12d0c" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/a925106a-7f18-486a-b169-8d704e29a270" />
</p>

Classification

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/507daeb3-35ee-4ac2-81ce-c71761a58e41" />
</p>

Why linear regression is not the best option

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/d4d28ded-5e08-470a-aeca-9ef29c89650c" />
</p>

Logistic Regression

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5f7bfa72-03e9-465b-94bc-ea0d6af4eea0" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/3d95ee21-13c3-4339-9e2e-7d1c275c91b8" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f3df5183-777c-4fa0-9d56-99e5828987e8" />
</p>

Decision Boundary

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/4437b03e-1e8b-4759-b1e7-fb8cfc57d6c6" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/14f1d720-d9e9-41bc-9c64-9e9a012800a6" />
</p>

<p><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/d1665553-da03-4c4e-aad1-d87085bb3c45" />
</p>

Cost Function for Logistic Regression












[Нагору ↑](#top)
