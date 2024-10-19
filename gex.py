import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, shapiro, linregress
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class DataAnalysis:
    def __init__(self):
        self.df = None
        self.column_types = {
            'interval': [],
            'nominal': []
        }

    def load_data(self, path):
        try:
            self.df = pd.read_csv(path)
            self.df.columns = [col.strip() for col in self.df.columns]
            self.update_column_types()
            print(f"Data loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def update_column_types(self):
        self.column_types['interval'] = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.column_types['nominal'] = self.df.select_dtypes(include=['object']).columns.tolist()

    def handle_missing_values(self):
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)
        choice = input("Choose how to handle missing values (1. Drop, 2. Fill with Mean, 3. Fill with Median): ")
        
        if choice == '1':
            self.df.dropna(inplace=True)
            print("Dropped rows with missing values.")
        elif choice == '2':
            for column in self.df.columns:
                if self.df[column].dtype in [np.float64, np.int64]:
                    mean_value = self.df[column].mean()
                    self.df[column].fillna(mean_value, inplace=True)
            print("Filled missing values with mean.")
        elif choice == '3':
            for column in self.df.columns:
                if self.df[column].dtype in [np.float64, np.int64]:
                    median_value = self.df[column].median()
                    self.df[column].fillna(median_value, inplace=True)
            print("Filled missing values with median.")
        else:
            print("Invalid choice.")

    def check_data_types(self):
        print("Data types of each column:")
        print(self.df.dtypes)

    def plot_histogram(self, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()

    def plot_bar_chart(self, column):
        counts = self.df[column].value_counts()
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title(f'Bar Chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def plot_scatterplot(self, x_column, y_column):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=x_column, y=y_column)
        plt.title(f'Scatterplot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def calculate_statistics(self, column):
        stats = self.df[column].describe()
        print(f"Statistics for {column}:")
        print(stats)

    def calculate_correlation(self):
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

    def calculate_standard_deviation(self, column):
        std_dev = self.df[column].std()
        print(f"Standard Deviation of {column}: {std_dev}")

    def calculate_kurtosis(self, column):
        kurt = self.df[column].kurtosis()
        print(f"Kurtosis of {column}: {kurt}")

    def calculate_skewness(self, column):
        skew = self.df[column].skew()
        print(f"Skewness of {column}: {skew}")

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
      """Choose t-test or Mann-Whitney U test based on normality."""
      groups = [group[continuous_var].dropna().values for name, group in self.df.groupby(categorical_var)]
    
    # Check if there are enough groups for the test
      if len(groups) < 2:
          print(f"Not enough groups for '{categorical_var}' to perform the test.")
          return
      
      # Check if groups have enough data
      if any(len(group) < 2 for group in groups):
          print("One or more groups do not have enough samples for the test.")
          return
      
      # Plot boxplot
      plt.figure(figsize=(10, 6))
      sns.boxplot(data=self.df, x=categorical_var, y=continuous_var)
      plt.title(f'Boxplot of {continuous_var} by {categorical_var}')
      plt.show()

    # Check normality for the first group
      try:
          stat, p_value = shapiro(groups[0])
          if p_value > 0.05:  # Normal distribution
              stat, p_value = ttest_ind(*groups)
              print(f"T-test results: Statistic = {stat:.4f}, p-value = {p_value:.15f}")
          else:  # Not normal distribution
              stat, p_value = mannwhitneyu(groups[0], groups[1])
              print(f"Mann-Whitney U test results: Statistic = {stat:.4f}, p-value = {p_value:.15f}")
      except Exception as e:
          print(f"Error during statistical tests: {e}")


    def chi_square_test(self):
        nominal_var1 = self.select_variable('nominal')
        nominal_var2 = self.select_variable('nominal')
        
        contingency_table = pd.crosstab(self.df[nominal_var1], self.df[nominal_var2])
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square test results: Statistic = {stat:.4f}, p-value = {p_value:.15f}")

        # Plotting the contingency table as a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
        plt.title(f'Contingency Table: {nominal_var1} vs {nominal_var2}')
        plt.xlabel(nominal_var2)
        plt.ylabel(nominal_var1)
        plt.show()

    def linear_regression(self):
        y_var = self.select_variable('interval')
        x_var = self.select_variable('interval')
        
        slope, intercept, r_value, p_value, std_err = linregress(self.df[x_var], self.df[y_var])
        print(f"Linear Regression results: Slope = {slope:.4f}, Intercept = {intercept:.4f}, R-squared = {r_value**2:.4f}")

        # Plotting the regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(x=self.df[x_var], y=self.df[y_var], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
        plt.title(f'Regression Line for {y_var} vs {x_var}')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

    def select_variable(self, data_type, max_categories=None):
        available_columns = [col for col in self.column_types[data_type] 
                             if max_categories is None or self.df[col].nunique() <= max_categories]
        
        if not available_columns:
            print(f"No available columns for {data_type} with max categories {max_categories}")
            return None
        
        print(f"Available {data_type} columns: {', '.join(available_columns)}")
        variable = input(f"Select a {data_type} variable: ")
        
        while variable not in available_columns:
            print("Invalid selection. Please try again.")
            variable = input(f"Select a {data_type} variable: ")
        
        return variable

class SentimentAnalysis:
    def __init__(self):
        self.df = None

    def load_data(self, path):
        try:
            self.df = pd.read_csv(path)
            print(f"Data loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = None

    def get_text_columns(self):
        text_columns = []
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                avg_length = self.df[column].str.len().mean()
                unique_entries = self.df[column].nunique()
                text_columns.append({'Column Name': column, 'Average Entry Length': avg_length, 'Unique Entries': unique_entries})
        return pd.DataFrame(text_columns)

    def vader_sentiment_analysis(self, data):
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []
        for text in data:
            if isinstance(text, str):
                score = analyzer.polarity_scores(text)['compound']
                scores.append(score)
                if score >= 0.05:
                    sentiments.append('positive')
                elif score <= -0.05:
                    sentiments.append('negative')
                else:
                    sentiments.append('neutral')
            else:
                scores.append(0)
                sentiments.append('unknown')
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        for text in data:
            if isinstance(text, str):
                score = TextBlob(text).sentiment.polarity
                scores.append(score)
                if score >= 0.05:
                    sentiments.append('positive')
                elif score <= -0.05:
                    sentiments.append('negative')
                else:
                    sentiments.append('neutral')
            else:
                scores.append(0)
                sentiments.append('unknown')
        return scores, sentiments

    def distilbert_sentiment_analysis(self, data):
        if pipeline is None:
            print("Transformers library not installed. Unable to use DistilBERT.")
            return [], []
        
        sentiment_pipeline = pipeline("sentiment-analysis")
        scores = []
        sentiments = []
        for text in data:
            if isinstance(text, str):
                result = sentiment_pipeline(text)[0]
                scores.append(result['score'] * (1 if result['label'] == 'POSITIVE' else -1))
                sentiments.append('positive' if result['label'] == 'POSITIVE' else 'negative')
            else:
                scores.append(0)
                sentiments.append('unknown')
        return scores, sentiments

    def main(self):
        text_columns_df = self.get_text_columns()
        print("Text Columns in Dataset:")
        print(text_columns_df)
        column_name = input("Enter the name of the text column to analyze: ")
        method = input("Choose the sentiment analysis method:\n1. VADER\n2. TextBlob\n3. DistilBERT\nEnter 1, 2, or 3: ")

        if method == '1':
            scores, sentiments = self.vader_sentiment_analysis(self.df[column_name])
        elif method == '2':
            scores, sentiments = self.textblob_sentiment_analysis(self.df[column_name])
        elif method == '3':
            scores, sentiments = self.distilbert_sentiment_analysis(self.df[column_name])
        else:
            print("Invalid method selected.")
            return

        self.df['sentiment_score'] = scores
        self.df['sentiment'] = sentiments
        print(self.df[['sentiment_score', 'sentiment']].head())

def main():
    print("Data Analysis and Sentiment Analysis Tool")
    print("1. Analyze Data")
    print("2. Perform Sentiment Analysis")
    print("3. Exit")
    choice = input("Choose an option (1, 2, or 3): ")

    path = input("Enter the path to the CSV file: ")

    if choice == '1':
        da = DataAnalysis()
        da.load_data(path)
        while True:
            print("\nData Analysis Options:")
            print("1. Handle Missing Values")
            print("2. Check Data Types")
            print("3. Plot Histogram")
            print("4. Plot Boxplot")
            print("5. Plot Bar Chart")
            print("6. Plot Scatterplot")
            print("7. Calculate Statistics")
            print("8. Calculate Correlation")
            print("9. Calculate Standard Deviation")
            print("10. Calculate Kurtosis")
            print("11. Calculate Skewness")
            print("12. Perform T-test or Mann-Whitney")
            print("13. Perform Chi-square Test")
            print("14. Perform Linear Regression")
            print("15. Exit")
            option = input("Choose an option (1-15): ")

            if option == '1':
                da.handle_missing_values()
            elif option == '2':
                da.check_data_types()
            elif option == '3':
                column = input("Enter the column name for the histogram: ")
                da.plot_histogram(column)
            elif option == '4':
                column = input("Enter the column name for the boxplot: ")
                da.plot_boxplot(column)
            elif option == '5':
                column = input("Enter the nominal column name for the bar chart: ")
                da.plot_bar_chart(column)
            elif option == '6':
                x_column = input("Enter the X variable name: ")
                y_column = input("Enter the Y variable name: ")
                da.plot_scatterplot(x_column, y_column)
            elif option == '7':
                column = input("Enter the column name for statistics: ")
                da.calculate_statistics(column)
            elif option == '8':
                da.calculate_correlation()
            elif option == '9':
                column = input("Enter the column name for standard deviation: ")
                da.calculate_standard_deviation(column)
            elif option == '10':
                column = input("Enter the column name for kurtosis: ")
                da.calculate_kurtosis(column)
            elif option == '11':
                column = input("Enter the column name for skewness: ")
                da.calculate_skewness(column)
            elif option == '12':
                continuous_var = da.select_variable('interval')
                categorical_var = da.select_variable('nominal')
                da.t_test_or_mannwhitney(continuous_var, categorical_var)
            elif option == '13':
                da.chi_square_test()
            elif option == '14':
                da.linear_regression()
            elif option == '15':
                print("Exiting Data Analysis.")
                break
            else:
                print("Invalid option. Please try again.")

    elif choice == '2':
        sa = SentimentAnalysis()
        sa.load_data(path)
        sa.main()
    elif choice == '3':
        print("Exiting.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
