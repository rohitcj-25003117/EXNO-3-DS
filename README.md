## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
             #-----------------------------------
      # FEATURE ENCODING
      #-----------------------------------
      
      
      
      import pandas as pd
      from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, QuantileTransformer
      from category_encoders import BinaryEncoder, TargetEncoder
      
      # Load your data
      df = pd.read_csv("/content/data.csv")
      print("Original Data:")
      print(df.head())
      
      #------------------------
      # Ordinal Encoding
      #------------------------
      
      # For Ord_1 (temperature-like categories)
      ord1_order = [['Cold', 'Warm', 'Hot', 'Very Hot']]
      e1 = OrdinalEncoder(categories=ord1_order)
      df['Ord_1_encoded'] = e1.fit_transform(df[['Ord_1']])
      print("\nAfter Ordinal Encoding (Ord_1):")
      print(df[['Ord_1', 'Ord_1_encoded']].head())
      
      # For Ord_2 (education levels)
      ord2_order = [['High School', 'Diploma', 'Bachelors', 'Masters', 'PhD']]
      e2 = OrdinalEncoder(categories=ord2_order)
      df['Ord_2_encoded'] = e2.fit_transform(df[['Ord_2']])
      print("\nAfter Ordinal Encoding (Ord_2):")
      print(df[['Ord_2', 'Ord_2_encoded']].head())
      
      
      #------------------------
      # Label Encoding
      #------------------------
      le = LabelEncoder()
      dfc = df.copy()
      dfc['City_encoded'] = le.fit_transform(dfc['City'])
      print("\nAfter Label Encoding (City):")
      print(dfc[['City', 'City_encoded']].head())
      
      
      #------------------------
      # OneHot Encoding
      #------------------------
      ohe = OneHotEncoder(sparse_output=False, drop=None)
      # keep all categories
      df2 = df.copy()
      encoded = pd.DataFrame(ohe.fit_transform(df2[['City']]),
                             columns=ohe.get_feature_names_out(['City']))
      df2 = pd.concat([df2, encoded], axis=1)
      print("\nAfter One-Hot Encoding:")
      print(df2.head())
      
      # Alternatively, using Pandas
      df2_dummies = pd.get_dummies(df.copy(), columns=['City'])
      print("\nUsing Pandas get_dummies:")
      print(df2_dummies.head())
      
      
      #------------------------
      # Binary Encoding
      #------------------------
      be = BinaryEncoder(cols=['Ord_2'])
      df_bin_be = be.fit_transform(df)
      print("\nAfter Binary Encoding:")
      print(df_bin_be.head())
      
      
      #------------------------
      # Target Encoding
      #------------------------
      te = TargetEncoder(cols=['Ord_2'])
      cc_te = te.fit_transform(df[['Ord_2']], df['Target'])
      cc = pd.concat([df, cc_te], axis=1)
      print("\nAfter Target Encoding:")
      print(cc.head())
      
      
      #-----------------------------------
      # FEATURE TRANSFORMATION
      #-----------------------------------
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns
      import statsmodels.api as sm
      import scipy.stats as stats
      
      df_t = pd.read_csv("/content/Data_to_Transform.csv")
      print("\nSkewness before transformation:")
      print(df_t.skew())
      
      # Log Transformation
      df_t["Log_Transform"] = np.log1p(df_t["Moderate Positive Skew"])
      
      # Reciprocal Transformation
      df_t["Reciprocal_Transform"] = 1 / (df_t["Moderate Positive Skew"] + 1)
      
      # Square Root Transformation
      df_t["Sqrt_Transform"] = np.sqrt(df_t["Moderate Positive Skew"])
      
      # Square Transformation
      df_t["Square_Transform"] = np.square(df_t["Moderate Positive Skew"])
      
      #------------------------
      # Box-Cox and Yeo-Johnson
      #------------------------
      # Box-Cox works only for strictly positive values
      df_t["BoxCox_Transform"], _ = stats.boxcox(df_t["Moderate Positive Skew"] + 1)
      
      # Yeo-Johnson works for both positive and negative values
      qt = QuantileTransformer(output_distribution='normal')
      df_t["YeoJohnson_Transform"] = qt.fit_transform(df_t[["Highly Negative Skew"]])
      
      print("\nSkewness after transformations:")
      print(df_t.skew())
      
      #------------------------
      # QQ Plots
      #------------------------
      sm.qqplot(df_t['Moderate Negative Skew'], line='45')
      plt.title("Before Transformation")
      plt.show()
      
      # Example after transformation
      df_t["Moderate Negative Skew_1"] = np.sqrt(df_t["Moderate Negative Skew"] + 10)  # shift to avoid negatives
      sm.qqplot(df_t['Moderate Negative Skew_1'], line='45')
      plt.title("After Transformation")
      plt.show()
      
      # Final Transformed Data
      print("\nFinal Transformed Data:")
      print(df_t.head())


<img width="920" height="649" alt="image" src="https://github.com/user-attachments/assets/65c567ef-6fee-4311-8afa-a491544ae4e6" />
<img width="976" height="608" alt="image" src="https://github.com/user-attachments/assets/b898e265-4acf-4c8a-ade3-632e62ff554e" />
<img width="940" height="615" alt="image" src="https://github.com/user-attachments/assets/ed5ea61e-0b55-4d1e-8bd2-24256c84d3e0" />
<img width="710" height="392" alt="image" src="https://github.com/user-attachments/assets/ea42f5c8-5a8b-495d-96ad-2916f58defac" />
<img width="825" height="765" alt="image" src="https://github.com/user-attachments/assets/6f6afb9c-b5ec-417c-9d76-35d1f3615b6b" />
<img width="1397" height="732" alt="image" src="https://github.com/user-attachments/assets/5c4078f1-e647-4ce7-8e03-176114dad8cc" />



# RESULT:
       Thus the expected output is achieved

       
