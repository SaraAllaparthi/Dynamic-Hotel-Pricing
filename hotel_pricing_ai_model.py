import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_data():
    """
    Load hotel bookings data from a CSV file located in the same folder.
    """
    file_path = os.path.join(os.path.dirname(__file__), "hotel_bookings.csv")
    try:
        df = pd.read_csv(file_path)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
        return df
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}")
        return None

def preprocess_occupancy_data(df):
    """
    Preprocess the data for Prophet:
      - Convert the 'reservation_status_date' column to datetime.
      - Filter out canceled bookings (where is_canceled == 0).
      - Group by date to compute daily occupancy.
      - Rename columns to 'ds' (date) and 'y' (value) for Prophet.
    """
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df_not_canceled = df[df['is_canceled'] == 0]
    occupancy = df_not_canceled.groupby('reservation_status_date').size().reset_index(name='occupancy')
    prophet_df = occupancy.rename(columns={'reservation_status_date': 'ds', 'occupancy': 'y'})
    return prophet_df

# -------------------------------
# Forecasting with Prophet
# -------------------------------

def fit_and_forecast(prophet_df, forecast_periods=30):
    """
    Initialize and fit the Prophet model, then generate a forecast.
    """
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    return model, forecast

def visualize_forecast(model, forecast):
    """
    Plot the forecast and its components.
    """
    fig1 = model.plot(forecast)
    plt.title("Occupancy Forecast")
    plt.xlabel("Date")
    plt.ylabel("Predicted Occupancy")
    st.pyplot(fig1)
    
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

def evaluate_forecast(model):
    """
    Evaluate forecast accuracy using Prophet's cross-validation.
    """
    df_cv = cross_validation(model, initial='90 days', period='15 days', horizon='30 days')
    df_p = performance_metrics(df_cv)
    st.write("Forecast Performance Metrics:")
    st.dataframe(df_p.head())

# -------------------------------
# Additional Feature Functions
# -------------------------------

def competitor_dynamic_pricing():
    st.header("Competitor-Based Dynamic Pricing")
    competitor_data = pd.DataFrame({
        'hotel': ['Competitor A', 'Competitor B', 'Competitor C'],
        'rate': [120.0, 135.0, 110.0],
        'occupancy_rate': [0.85, 0.90, 0.80]
    })
    st.write("Competitor Data:")
    st.dataframe(competitor_data)
    avg_competitor_rate = competitor_data['rate'].mean()
    avg_competitor_occ = competitor_data['occupancy_rate'].mean()
    adjustment_factor = 1 + (avg_competitor_occ - 0.85)  # Dummy adjustment
    recommended_price = avg_competitor_rate * adjustment_factor
    st.write(f"Recommended Base Price based on competitors: ${recommended_price:.2f}")

def demand_forecasting_with_events(prophet_df):
    st.header("Demand Forecasting with Events")
    # Hardcoded event/holiday dates for demonstration
    event_dates = ['2015-07-04', '2015-12-25']
    holidays = pd.DataFrame({
        'holiday': 'event',
        'ds': pd.to_datetime(event_dates),
        'lower_window': 0,
        'upper_window': 1,
    })
    st.write("Events/Holidays for Forecast:")
    st.dataframe(holidays)
    
    model = Prophet(holidays=holidays, daily_seasonality=True)
    model.fit(prophet_df)
    forecast_periods = 30
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    st.write("Forecast Data (last 5 rows):")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig = model.plot(forecast)
    plt.title("Demand Forecast with Events")
    plt.xlabel("Date")
    plt.ylabel("Predicted Occupancy")
    st.pyplot(fig)

def personalized_promotions():
    st.header("Personalized Promotions")
    customer_data = pd.DataFrame({
        'customer_type': ['Transient', 'Contract', 'Group', 'Transient', 'Transient'],
        'average_spend': [100, 150, 80, 120, 90]
    })
    st.write("Customer Data Sample:")
    st.dataframe(customer_data)
    promotions = {
        'Transient': 'Offer 10% off for early bookings',
        'Contract': 'Provide a complimentary room upgrade',
        'Group': 'Offer special group discounts'
    }
    chosen_segment = 'Transient'  # This could be made interactive
    st.write(f"Suggested promotion for {chosen_segment} customers: {promotions[chosen_segment]}")

def los_optimization():
    st.header("Length-of-Stay (LOS) Optimization")
    los_data = pd.DataFrame({
        'LOS (nights)': [1, 2, 3, 4, 5, 6],
        'average_rate': [100, 95, 90, 85, 80, 75]
    })
    st.write("Historical LOS Data:")
    st.dataframe(los_data)
    optimal_index = los_data['average_rate'].idxmax()
    optimal_los = los_data.loc[optimal_index, 'LOS (nights)']
    st.write(f"Optimal Length of Stay: {optimal_los} nights")
    st.write("Consider incentivizing this LOS with special pricing or packages.")

def room_upgrade_suggestions():
    st.header("Room Upgrade and Cross-Selling Suggestions")
    booking_data = pd.DataFrame({
        'guest_id': [1, 2, 3, 4, 5],
        'current_room_type': ['Standard', 'Standard', 'Deluxe', 'Standard', 'Suite'],
        'past_upgrade': [False, True, False, False, True]
    })
    st.write("Booking Data:")
    st.dataframe(booking_data)
    suggestions = booking_data[booking_data['past_upgrade'] == False]
    st.write("Room Upgrade Suggestions for Guests:")
    st.dataframe(suggestions[['guest_id', 'current_room_type']])
    st.write("Consider offering packages (e.g., breakfast or spa) to encourage upgrades.")

def price_sensitivity_analysis():
    st.header("Price Sensitivity Analysis")
    pricing_data = pd.DataFrame({
        'price': [80, 90, 100, 110, 120, 130],
        'conversion_rate': [0.9, 0.88, 0.85, 0.8, 0.75, 0.7]
    })
    st.write("Pricing and Conversion Data:")
    st.dataframe(pricing_data)
    threshold = pricing_data[pricing_data['conversion_rate'] < 0.8]['price'].min()
    if pd.isna(threshold):
        st.write("No significant drop in conversion rate detected.")
    else:
        st.write(f"Price sensitivity threshold detected at: ${threshold}")
        st.write("Consider setting room rates below this threshold to maintain bookings.")

def last_minute_pricing():
    st.header("Last-Minute Pricing Adjustments")
    unsold_inventory = 5  # Example value
    current_price = 100.0
    st.write(f"Current unsold inventory: {unsold_inventory} rooms")
    st.write(f"Current base price: ${current_price}")
    if unsold_inventory > 0:
        adjusted_price = current_price * 0.9
        st.write(f"Suggested last-minute price adjustment: ${adjusted_price:.2f}")
    else:
        st.write("No adjustments needed for last-minute pricing.")

def competitor_benchmarking_dashboard():
    st.header("Competitor Benchmarking Dashboard")
    competitor_data = pd.DataFrame({
        'hotel': ['Competitor A', 'Competitor B', 'Competitor C', 'Our Hotel'],
        'price': [120, 135, 110, 125],
        'occupancy_rate': [0.85, 0.90, 0.80, 0.88]
    })
    st.write("Competitor Benchmarking Data:")
    st.dataframe(competitor_data)
    fig, ax = plt.subplots()
    ax.bar(competitor_data['hotel'], competitor_data['price'], color=['blue', 'green', 'red', 'purple'])
    ax.set_ylabel("Room Price")
    ax.set_title("Competitor Room Prices")
    st.pyplot(fig)

# -------------------------------
# Main Application
# -------------------------------

def main():
    st.title("Hotel Pricing AI Model")
    
    # Load the data
    df = load_data()
    if df is None:
        return
    
    st.header("Preprocessing Data")
    prophet_df = preprocess_occupancy_data(df)
    st.write("Processed occupancy data (first 5 rows):")
    st.dataframe(prophet_df.head())
    
    st.header("Prophet Forecasting Model")
    model, forecast = fit_and_forecast(prophet_df)
    st.write("Forecast Data (tail):")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    visualize_forecast(model, forecast)
    
    st.header("Forecast Accuracy Evaluation")
    evaluate_forecast(model)
    
    # Additional Features accessible via sidebar
    st.sidebar.title("Additional Features")
    feature = st.sidebar.selectbox("Select Feature", 
                                   ["Competitor-Based Dynamic Pricing",
                                    "Demand Forecasting with Events",
                                    "Personalized Promotions",
                                    "LOS Optimization",
                                    "Room Upgrade Suggestions",
                                    "Price Sensitivity Analysis",
                                    "Last-Minute Pricing Adjustments",
                                    "Competitor Benchmarking Dashboard"])
    
    if feature == "Competitor-Based Dynamic Pricing":
        competitor_dynamic_pricing()
    elif feature == "Demand Forecasting with Events":
        demand_forecasting_with_events(prophet_df)
    elif feature == "Personalized Promotions":
        personalized_promotions()
    elif feature == "LOS Optimization":
        los_optimization()
    elif feature == "Room Upgrade Suggestions":
        room_upgrade_suggestions()
    elif feature == "Price Sensitivity Analysis":
        price_sensitivity_analysis()
    elif feature == "Last-Minute Pricing Adjustments":
        last_minute_pricing()
    elif feature == "Competitor Benchmarking Dashboard":
        competitor_benchmarking_dashboard()

if __name__ == '__main__':
    main()
