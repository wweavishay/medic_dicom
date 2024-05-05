import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table


def create_connection(server_name, db_name):
    conn_str = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={db_name};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str)

def fetch_data():
    server_name = "DESKTOP-4VV2GQ3"
    db_name = "medicdb"
    conn = create_connection(server_name, db_name)
    cursor = conn.cursor()
    cursor.execute("""SELECT PatientIdentifier, PatientName, Sex, Age, TypeofDisease, StudyDescription, ProtocolName, BodyPartExamined
                    FROM (
                        SELECT 
                            pi.PatientID AS PatientIdentifier,
                            pi.PatientName,
                            pi.Sex,
                            pi.Age,
                            dm.TypeofDisease,
                            dm.StudyDescription,
                            dm.ProtocolName,
                            dm.BodyPartExamined,
                            ROW_NUMBER() OVER (PARTITION BY pi.PatientID ORDER BY pi.PatientID) AS RowNum
                        FROM 
                            [medicdb].[dbo].[PatientInformation] pi
                        INNER JOIN 
                            [medicdb].[dbo].[DICOMMetadata] dm ON pi.PatientID = dm.PatientID
                    ) AS Subquery
                    WHERE RowNum = 1;""")
    rows = cursor.fetchall()
    conn.close()
    return rows

def read_data(data):
    columns = ['PatientName', 'PatientID', 'Sex', 'Age', 'foldername', 'StudyDescription', 'ProtocolName', 'BodyPartExamined']
    df = pd.DataFrame(columns=columns)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if i >= len(df):
                df.loc[i] = [None] * len(columns)
            if j >= len(df.columns):
                df[df.columns[j]] = [None] * len(df)
            df.iloc[i, j] = data[i][j]
    return df

def plot_data(data):
    # Create subplots for the first set of plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    # Plot Gender Distribution
    data['Sex'].value_counts().plot(kind='bar', color=['lightblue', 'lightpink'], alpha=0.7, ax=axes1[0, 0])
    axes1[0, 0].set_title('Gender Distribution')
    axes1[0, 0].set_xlabel('Gender')
    axes1[0, 0].set_ylabel('Count')
    axes1[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Check if "age" key exists in data
    if "age" in data:
        try:
            # Convert age data to integer
            age = int(data["age"])
        except (ValueError, TypeError):
            # If conversion fails, set age to 0
            age = 0
    else:
        # If "age" key doesn't exist, set age to 0
        age = 0

    data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(0).astype(int)
    filtered_age = [age for age in data['Age'] if age is not None]
    # Sort filtered age data
    sorted_age = sorted(filtered_age)

    # Plot Age Distribution
    axes1[0, 1].hist(sorted_age, bins=20, color='lightgreen', alpha=0.7)
    axes1[0, 1].set_title('Age Distribution')
    axes1[0, 1].set_xlabel('Age')
    axes1[0, 1].set_ylabel('Frequency')
    axes1[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Study Description Distribution
    data['StudyDescription'].value_counts().head(10).plot(kind='bar', color='lightcoral', alpha=0.7, ax=axes1[1, 0])
    axes1[1, 0].set_title('Top 10 Study Descriptions')
    axes1[1, 0].set_xlabel('Study Description')
    axes1[1, 0].set_ylabel('Count')
    axes1[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    # Plot Protocol Name Distribution
    data['ProtocolName'].value_counts().head(10).plot(kind='bar', color='lightskyblue', alpha=0.7, ax=axes1[1, 1])
    axes1[1, 1].set_title('Top 10 Protocol Names')
    axes1[1, 1].set_xlabel('Protocol Name')
    axes1[1, 1].set_ylabel('Count')
    axes1[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust layout for the first set of plots
    plt.tight_layout()
    # Save the first set of plots as an image
    plt.savefig('plots/plots_part1.png')
    # Show the first set of plots
    plt.show()

    # Create subplots for the second set of plots
    fig2, axes2 = plt.subplots(1, 1, figsize=(6, 5))
    # Plot Body Part Examined Distribution
    data['BodyPartExamined'].value_counts().head(10).plot(kind='bar', color='lightskyblue', alpha=0.7, ax=axes2)
    axes2.set_title('Top 10 Body Parts Examined')
    axes2.set_xlabel('Body Part Examined')
    axes2.set_ylabel('Count')
    axes2.grid(axis='y', linestyle='--', alpha=0.7)
    # Adjust layout for the second set of plots
    plt.tight_layout()
    # Save the second set of plots as an image
    plt.savefig('plots/plots_part2.png')
    # Show the second set of plots
    plt.show()

def display_statistics(data):
    statistics = data.describe(include='all').transpose()
    print("Statistics for all parameters:")
    print(statistics)

def write_to_excel(data, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Write DataFrame to Excel
        data.to_excel(writer, sheet_name='Data', index=False)
        # Write statistics to Excel
        statistics = data.describe(include='all').transpose()
        statistics.to_excel(writer, sheet_name='Statistics')
        # Create a new sheet for each parameter and plot distribution
        for column in data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            if data[column].dtype == 'object':
                # Plot bar chart for categorical data
                data[column].value_counts().plot(kind='bar', ax=ax)
            else:
                # Plot histogram for numerical data
                data[column].plot(kind='hist', bins=20, ax=ax)
            plt.title(f'{column} Distribution')
            plt.xlabel(column)
            plt.ylabel('Count' if data[column].dtype == 'object' else 'Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'plots/{column}_distribution.png')
            plt.close(fig)
            # Add plot to Excel
            img_sheet = writer.book.add_worksheet(f'{column}_distribution')
            img_sheet.insert_image(0, 0, f'plots/{column}_distribution.png')

def main():
    print("Main Menu")
    print("1. Fetch Data")
    print("2. Read Data")
    print("3. Analyze and Plot Data")
    print("4. View Statistics")
    print("5. Exit")
    while True:
        choice = input("Enter your choice: ")
        if choice == "1":
            data = fetch_data()
            print("Data fetched successfully.")
        elif choice == "2":
            if 'data' not in locals():
                print("Please fetch the data first.")
                continue
            data = read_data(data)
            print("Data read successfully.")
        elif choice == "3":
            if 'data' not in locals():
                print("Please fetch and read the data first.")
                continue
            plot_data(data)
            write_to_excel(data, 'outputplot/analysis_results.xlsx')
        elif choice == "4":
            if 'data' not in locals():
                print("Please fetch and read the data first.")
                continue
            display_statistics(data)
        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
