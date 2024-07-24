import json
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode

# Enable Plotly and Cufflinks offline mode
cf.go_offline(connected=True)
init_notebook_mode(connected=True)


def clear_directory(directory, extension=".png"):
    """Delete all files with a given extension in the directory."""
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted existing plot: {file_path}")

def plot_attribute(input_file_path, plot_file_path):
    # Ensure the plot_file_path directory exists; create if not
    if not os.path.exists(plot_file_path):
        os.makedirs(plot_file_path)

    # Clear the directory of existing .png files
    clear_directory(plot_file_path, ".png")

    # Iterate through each JSON file in input_file_path directory
    for filename in os.listdir(input_file_path):
        if filename.endswith(".json"):
            # Check if the file starts with any attribute in the list
            input_file = os.path.join(input_file_path, filename)
            plot_file = os.path.join(plot_file_path, os.path.splitext(filename)[0] + ".png")    

            # Check if the plot file exists and delete it if it does
            if os.path.exists(plot_file):
                os.remove(plot_file)
                print(f"Existing plot {plot_file} deleted.")    

            try:
                # read the JSON  file 
                with open(input_file, "r") as f:
                    data = json.load(f)

                # convert JSON data to DataFrame
                records = []
                for entry in data:
                    timestamp = entry["timestamp"]
                    attribute_name = entry["attribute_name"]
                    for key, value in entry["values"].items():
                        records.append({
                            "timestamp": timestamp,
                            "attribute_name": attribute_name,
                            "variable": key,
                            "value": value
                        })

                df = pd.DataFrame(records)
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Calculate time difference in seconds
                df["time_diff"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

                # Generate plots for q and q_d
                df_q_qd = df[df['attribute_name'].isin(["q", "q_d"])]
                fig_q_qd = px.line(df_q_qd, x='time_diff', y='value', color='variable')
                fig_q_qd.update_layout(
                    width=1400, height=600,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=5),
                    hovermode='x unified'
                )
                plot_file = os.path.join(plot_file_path, "q & q_d.png")
                fig_q_qd.write_image(plot_file)
                print(f"Plot saved: {plot_file}")

                # Generate plots for tau_ext_hat_filtered and tau_J
                df_tau_ext = df[df['attribute_name'] == "tau_ext_hat_filtered"]
                df_tau_J = df[df['attribute_name'] == "tau_J"]

                combined_df_tau = pd.concat([df_tau_ext, df_tau_J]).pivot(
                    index="time_diff", columns=["attribute_name", "variable"], values="value"
                )

                # Generate combined time series plot for tau_ext_hat_filtered and tau_J
                combined_plot = combined_df_tau.iplot(
                    xTitle="time", yTitle="value", asFigure=True
                )
                combined_plot.update_layout(
                    width=1400, height=600,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=5),
                    hovermode='x unified'
                )
                tau_combined_plot_file = os.path.join(plot_file_path, "tau_combined.png")
                pio.write_image(combined_plot, tau_combined_plot_file)
                print(f"Plot saved: {tau_combined_plot_file}")

                # Generate individual time series plots for each combination of variables
                tau_ext_vars = df_tau_ext['variable'].unique()
                tau_J_vars = df_tau_J['variable'].unique()

                # Pair variables by their index
                min_length = min(len(tau_ext_vars), len(tau_J_vars))
                for i in range(min_length):
                    tau_ext_var = tau_ext_vars[i]
                    tau_J_var = tau_J_vars[i]

                    df_tau_ext_var = df_tau_ext[df_tau_ext['variable'] == tau_ext_var]
                    df_tau_J_var = df_tau_J[df_tau_J['variable'] == tau_J_var]

                    # Combine both variable data for plotting
                    df_combined_var = pd.concat([df_tau_ext_var, df_tau_J_var])
                    if not df_combined_var.empty:
                        fig_combined_var = px.line(df_combined_var, x='time_diff', y='value', color='attribute_name', title=f'{tau_ext_var} vs {tau_J_var}')
                        fig_combined_var.update_layout(
                            width=1400, height=600,
                            xaxis=dict(tickmode='linear', tick0=0, dtick=5),
                            hovermode='x unified'
                        )
                        var_plot_file = os.path.join(plot_file_path, f"{tau_ext_var}_{tau_J_var}.png")
                        fig_combined_var.write_image(var_plot_file)
                        print(f"Plot saved: {var_plot_file}")

            except Exception as e:
                print(f"Failed to process file {input_file}: {e}")


if __name__ == '__main__':
    input_file_path = "data/0123"
    plot_file_path = "data/0123/plot"
    plot_attribute(input_file_path, plot_file_path)
