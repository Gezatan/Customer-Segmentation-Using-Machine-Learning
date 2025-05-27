# Customer-Segmentation-Using-Machine-Learning
Developed a Machine Learning code for creating customer segmentation (Cluster Analysis, Elbow Point, Insights) 

This project applies K-Means clustering to segment customers based on their annual income and spending score, enabling the product team to design tailored strategies for each segment (e.g., VIP users, bargain hunters, etc.).

## Features
- **Data Preprocessing:** Reads customer data from a CSV file.
- **Elbow Method:** Determines the optimal number of clusters.
- **K-Means Clustering:** Groups customers into distinct segments.
- **Visualization:** Scatter plot of customer segments (X: Spending Score, Y: Annual Income).
- **Cluster Summary:** Table and insights for each segment.

## How to Run
1. Place your `Mall_Customers.csv` file in the project directory.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Run the script:
   ```bash
   python Customer_Segmentation.py
   ```

## Output
- **Elbow Plot:** Helps choose the optimal number of clusters.
- **Scatter Plot:** Visualizes customer segments.
- **Summary Table:** Shows key statistics for each cluster.
- **Insights:** Interprets each segment (e.g., VIP, bargain hunters).

## Columns Used
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

