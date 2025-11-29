# InFoAnalytics: AI for Sustainable Development

**Predicting Fiscal Instability to Secure SDG Funding**

InFoAnalytics is an advanced AI-powered dashboard designed to transform fragmented macroeconomic data into actionable insights. By predicting fiscal shifts with high accuracy, we enable policy-makers to implement proactive buffers that protect critical development goals, specifically **SDG 3 (Health)** and **SDG 4 (Education)**, which are often the first casualties of economic stress.

## 📸 Visual Overview

### Main Dashboard
![Main Content](1.png)
*Real-time economic intelligence and predictive analytics at a glance.*

### Strategic Insights
![Macroeconomic Correlations & Volatility](2.png)
*Analyzing dependencies between economic indicators and assessing volatility risks.*

### Unemployment Trends
![Unemployment Trends](3.png)
*Comparative analysis of unemployment rates across different regions.*

### Market Analysis
![Market Analysis](4.png)
*Comprehensive view of fiscal performance, revenue trends, and economic health scores.*

---

## 🏆 Hackathon Strategy: Addressing the 4 Pillars

This project is strategically engineered to address the four key judging criteria of the **10Alytics Global Data Hackathon**.

### 1. Understandability (Problem Framing)
**"Predicting Fiscal Instability to Secure SDG Funding"**

*   **The Problem:** Budget volatility jeopardizes critical Sustainable Development Goals. When deficits swing unpredictably, funding for Health (SDG 3) and Education (SDG 4) is often cut first.
*   **The Scope:** We acknowledge the complexity of African economies—multiple currencies, unique contexts, and diverse indicators. Our **Grouped Long-Format Approach** respects these unique national contexts.
*   **The Why:** The challenge isn't just knowing *what* the deficit is, but *why* policy-makers can't predict its change. We enable them to foresee these shifts.

### 2. Innovativeness (Quality of Insights)
**Advanced Machine Learning Architecture**

*   **Rate of Change (Differencing):** We moved beyond simple value prediction. By training our model on the *rate of change* (differencing), we achieved a massive leap in forecasting stability and complexity.
*   **Feature Engineering:** Our model combines **Grouped Lag** and **Rolling Mean** features with **One-Hot Encoded Country** data. This allows the AI to learn from each nation's unique economic history.
*   **Key Insight:** The model reveals that **Value_lag_1** (the previous period's momentum) is the dominant driver (**78.5%** importance). This proves that *fiscal momentum* is the single most critical factor for predictability.

### 3. Impactfulness (Completeness of Solution)
**From Data to Policy Action**

We translate technical insights into concrete policy levers:
*   **Insight:** Fiscal Momentum (78.5%) is the key driver.
*   **Recommendation:** Implement a **"Fiscal Momentum Buffer"**. Policy-makers should mandate reserve accumulation when the deficit trends positively for two consecutive quarters to mitigate short-term risks.
*   **Insight:** Country-specific context matters.
*   **Recommendation:** Use our **Dynamic Dashboard** for country-specific risk assessments, rejecting a "one-size-fits-all" approach for Africa.
*   **Future Outlook:** Our model achieves an **R² of 0.8660**, providing confident forecasts for the next 1-3 periods.

### 4. Applicability (Enthusiasm & Presentation)
**Visual Clarity & Confidence**

*   **Clean Visualization:** Our dashboard presents complex data—like Budget Deficit Predictions and Top Feature Importance—in highly legible, interactive charts.
*   **Confidence in Complexity:** We simplify the complex. We "taught the AI to look at the rate of change because the absolute numbers were too noisy for decision-making."

---

## 🛠️ Technology Stack

*   **Backend:** Python, Flask
*   **AI/ML:** Scikit-learn, Gradient Boosting Regressor (R² = 0.8660)
*   **Data Processing:** Pandas, NumPy (Grouped Lag/Rolling Features)
*   **Frontend:** HTML5, Tailwind CSS, JavaScript
*   **Visualization:** amCharts 5, Matplotlib, Seaborn

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd InfoAnalytics
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    flask run
    ```
    Or for external access:
    ```bash
    flask run --host 0.0.0.0
    ```

4.  **Access the dashboard:**
    Open your browser and navigate to `http://127.0.0.1:5000`.

---
*Developed by Leon Motaung for the 10Alytics Global Data Hackathon*
