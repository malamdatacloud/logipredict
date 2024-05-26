import shap
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
shap.initjs()

from user_auth import UserAuthentication
from air.air import predict_top_companies, process_and_visualize
from ocean.ocean import predict_top_companies, process_and_visualize


with open("C:/Users/User/Desktop/Projects/Fritz/CopyPrediApp/routes_air.json") as f:
    routes_air = json.load(f)

with open("C:/Users/User/Desktop/Projects/Fritz/CopyPrediApp/routes_ocean.json") as f:
    routes_ocean = json.load(f)

def main():
    st.title("Welcome to LogiPredict!")
    st.image("C:/Users/User/Desktop/Projects/Fritz/CopyPrediApp/background.webp",
             caption="LogiPredict",
             use_column_width=True)

    # Initialize the User Auth Class Object
    auth = UserAuthentication()
    st.sidebar.title("User Authentication")

    # Check if the user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    # Track the current authentication mode
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = None

    if st.session_state.logged_in:
        display_main_content(auth, st.session_state.username)
        if st.sidebar.button("View Saved Results"):
            saved_results = auth.get_user_results(st.session_state.username)
            st.sidebar.write("Saved Results:")
            for i, result in enumerate(saved_results):
                st.sidebar.write(f"Result {i+1}:")
                st.sidebar.dataframe(result)
                csv = result.to_csv(index=False).encode()
                st.sidebar.download_button(
                    label=f"Download Result {i+1} as CSV",
                    data=csv,
                    file_name=f'result_{i+1}.csv',
                    mime='text/csv',
                )
    else:
        login_button = st.sidebar.button("Log In", key="login_button")
        signup_button = st.sidebar.button("Sign Up", key="signup_button")

        if login_button:
            st.session_state.auth_mode = "Log In"
        if signup_button:
            st.session_state.auth_mode = "Sign Up"

        if st.session_state.auth_mode == "Log In":
            username_login = st.sidebar.text_input("Username", key='username_login')
            password_login = st.sidebar.text_input("Password", type="password", key='password_login')
            if st.sidebar.button("Sign In", key="signin_button"):
                if auth.log_in(username_login, password_login):
                    st.sidebar.success("Successfully logged in!")
                    st.session_state.logged_in = True
                    st.session_state.username = username_login
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Invalid Username or Password")

        if st.session_state.auth_mode == "Sign Up":
            first_name = st.sidebar.text_input("First Name", key='first_name')
            last_name = st.sidebar.text_input("Last Name", key='last_name')
            email = st.sidebar.text_input("Email", key='email')
            username_sign_up = st.sidebar.text_input("Username", key='username_sign_up')
            password_sign_up = st.sidebar.text_input("Password", type="password", key='password_sign_up')
            st.sidebar.write("Password must contain 1 uppercase, 1 lowercase, 1 number and one of these symbols: !@#$%^&")

            if st.sidebar.button("Sign Up", key="signup_form_button"):
                if auth.sign_up(first_name, last_name, email, username_sign_up, password_sign_up):
                    st.sidebar.success("Sign up successful! Please Log In to continue")
                    st.session_state.auth_mode = 'Log In'
                else:
                    st.sidebar.error("Sign up failed, Username may already be taken or password does not meet requirements.")

# Set Title for the APP
def display_main_content(auth, username):
    st.title('LogiPredict: Smart Shipping Solutions')

    # Dropdown for choosing between Air and Ocean
    transport_type = st.selectbox(
        'Choose your shipping transport method: ',
        ('Air', 'Ocean')
    )

    # Define dynamic lists for loading and destination options based on the shipping method
    if transport_type == 'Air':

        loading_ports = (
            'AMSTERDAM',
            'Dallas,tx',
            'FRANKFURT',
            'Houston,tx',
            'MANCHESTER',
            'Miami,fl',
            'SHANGHAI',
            'Stuttgart',
            'TORONTO',
            'Tel-Aviv'
        )

        loading_countries = (
            'CANADA',
            'CHINA',
            'GERMANY',
            'ISRAEL',
            'NETHERLANDS',
            'UNITED KINGDOM',
            'USA'
        )

        destination_ports = (
            'ABIDJAN',
    'ABU DHABI',
    'ANTALYA',
    'ASUNCION',
    'AUCKLAND',
    'Abuja',
    'Accra',
    'Adana',
    'Addis Abeba',
    'Ahmedabad',
    'Alma Ata',
    'Amsterdam',
    'Ankara-Esenboga',
    'Antananarivo (Tanannarive)',
    'Antigua',
    'Ataturk',
    'Athens',
    'Atlanta,ga',
    'Austin,tx',
    'BAMAKO',
    'BANGKOK',
    'BARCELONA',
    'BRISBANE',
    'BUENOS AIRES',
    'Baku',
    'Bangalore',
    'Beijing',
    'Belgrad(Beograd)',
    'Belize City',
    'Berlin',
    'Berlin-Tegel',
    'Billund',
    'Bishkek',
    'Bogota',
    'Boise,id',
    'Bologna',
    'Bombay - Mumbai',
    'Boston,ma',
    'Bratislava',
    'Bridgetown',
    'Brussels',
    'Budapest',
    'Buenos Aires-ezeza/Ministro P',
    'Bukarest-Otopeni',
    'CAPE TOWN',
    'CASABLANCA',
    'COCHIN',
    'COLOMBO',
    'Cairo',
    'Calgary/Banff',
    'Casablanca-Mohamed',
    'Charlotte,nc',
    'Chengdu',
    'Chennai ( Madras )',
    'Chicago Ohare,il',
    'Cincinnati,oh',
    'Cleveland Hopkins Internationa',
    'Cologne (Koeln) / Bonn',
    'Columbus,oh',
    'Conarkry',
    'Copenhagen',
    'Curitiba,pr',
    'DAKAR',
    'DIASS',
    'DUBLIN',
    'DURBAN',
    'Dalian',
    'Dallas,tx',
    'Dar Es Salam (Daressalam)',
    'Delhi',
    'Denver,co',
    'Detroit,mi',
    'Dresden',
    'Dubai',
    'Duesseldorf',
    'Dushanbe (Duschanbe)',
    'East Midlands',
    'Eindhoven',
    'El Paso,tx',
    'Entebbe',
    'Ercan',
    'Fort De France',
    'Frankfurt',
    'GENOA',
    'GOTEBORG',
    'GUATEMALA CITY',
    'GUAYAQUIL',
    'Gaborone',
    'Geneva',
    'Grand Rapids,mi',
    'Guadalajara',
    'Guangzhou (kanton)',
    'HAMBURG',
    'HONG KONG',
    'Hannover',
    'Hanoi',
    'Harare',
    'Havanna',
    'Helsinki',
    'Ho Chi Minh City (Saigon)',
    'Houston,tx',
    'Huntsville, AL',
    'Hyderabad',
    'ISTANBUL',
    'Jakarta - Kemayoran',
    'Jakarta - Sukarno Hatta',
    'Johannesburg',
    'KINGSTON',
    'Kathmandu',
    'Kiev-Borispol',
    'Kigali',
    'Kilimanjaro',
    'Kinshasa',
    'Kishinev',
    'Krasnodar',
    'Kuala Lumpur',
    'LIBREVILLE',
    'LISBON',
    'Lagos',
    'Lanzhou',
    'Larnaca',
    'Las Vegas,nv',
    'Liege',
    'Lilongwe',
    'Lima',
    'Ljubljana',
    'London-Heathrow',
    'Los Angeles,ca',
    'Louisville,ky',
    'Luqa (Valletta)',
    'Lusaka',
    'Luxemburg',
    'Lyon',
    'MELBOURNE',
    'MONTEVIDEO',
    'MONTREAL',
    'MUNICH',
    'Maastricht',
    'Madrid',
    'Mahe',
    'Malaga',
    'Malmoe-Sturup',
    'Managua',
    'Manchester',
    'Manila',
    'Mauritius',
    'Mcallen,tx',
    'Medellin',
    'Memphis, tn',
    'Mendoza',
    'Mexico City',
    'Miami,fl',
    'Milan',
    'Milan - Malpensa',
    'Minneapolis,mn',
    'Minsk',
    'Mombassa',
    'Montreal-Dorval',
    'Moscow - Domodedovo',
    'Moscow-Sheremetyevo 1',
    'Moscow-Vnukovo',
    'NAGOYA',
    'NEW YORK, NY',
    'Nairobi',
    'Nanjing',
    'Newark, NJ',
    'ODESSA',
    'OSAKA',
    'OSLO',
    'Oporto',
    'Orlando,fl',
    'Osaka - Kansai Airport',
    'Ougadougou',
    'PALERMO',
    'PANAMA CITY',
    'Paris-Charles de Gaulle',
    'Perth',
    'Philadelphia,pa',
    'Phoenix,az',
    'Pittsburgh,pa',
    'Port Moresby',
    'Port of Spain',
    'Portland,or',
    'Prague',
    'Pristina',
    'Qingdao',
    'Quito',
    'Richmond,va',
    'Riga',
    'Roberts International Airport',
    'Rome-Leonardo Da Vinci/Fuimici',
    'Roterdam',
    'Rzeszow',
    'SAN SALVADOR',
    'SHANGHAI',
    'SINGAPORE',
    'SURABAYA',
    'SYDNEY',
    'Sal',
    'Saloniki',
    'Salt Lake City,ut',
    'San Diego,ca',
    'San Francisco,ca',
    'San Jose',
    'San Juan,pr',
    'San Pedro Sula',
    'Santiago De Chile',
    'Santo Domingo',
    'Sao Paulo-Guarulhos',
    'Sao Paulo-Viracopos',
    'Seattle, WA',
    'Seoul-Incheon Airport',
    'Shannon (Limerick)',
    'Shenzhen',
    'Skopje',
    'Sofia',
    "St' Louis, MO",
    'St. Petersburg (Leningrad)',
    'Stockholm',
    'Stuttgart',
    'TORONTO',
    'Taipei-Chiang Kaisek',
    'Tallinn',
    'Tampa, FL',
    'Tashkent',
    'Tbilisi',
    'Teneriff South',
    'Tirana',
    'Tokyo - Narita Airport',
    'VALENCIA',
    'VIENNA',
    'Vilnius',
    'Viru Viru Airport',
    'Volgograd',
    'Warschau',
    'Washington, DC',
    'Winnipeg',
    'Wuhan',
    'Yerevan',
    'Zagreb',
    'Zuerich (Zurich)'
        )

        destination_countries = (
            'ALBANIA',
    'ANTIGUA And BARBUDA',
    'ARGENTINA',
    'ARMENIA',
    'AUSTRALIA',
    'AUSTRIA',
    'AZARBEIJAN',
    'BARBADOS',
    'BELARUS',
    'BELGIUM',
    'BELIZE',
    'BOLIVIA',
    'BOTSWANA',
    'BRAZIL',
    'BULGARIA',
    'BURKINA FASO',
    'CANADA',
    'CAPE VERDE',
    'CHILE',
    'CHINA',
    'COLOMBIA',
    'COSTA RICA',
    'CROATIA',
    'CUBA',
    'CYPRUS',
    'CZECH REPUBLIC',
    'DENMARK',
    'DOMINICAN REPUBLIC',
    'ECUADOR',
    'EGYPT',
    'EL SALVADOR',
    'ESTONIA',
    'ETHIOPIA',
    'FINLAND',
    'FRANCE',
    'GABON',
    'GEORGIA',
    'GERMANY',
    'GHANA',
    'GREECE',
    'GUATEMALA',
    'GUINEA BISSAU',
    'HONDURAS',
    'HONG KONG',
    'HUNGARY',
    'INDIA',
    'INDONESIA',
    'IRELAND',
    'ITALY',
    'IVORY COAST',
    'JAMAICA',
    'JAPAN',
    'KAZAKHSTAN',
    'KENYA',
    'KOREA - south',
    'KYRGUSTAN',
    'LATVIA',
    'LIBERIA',
    'LITHUANIA',
    'LUXEMBOURG',
    'MACEDONIA',
    'MADAGASCAR',
    'MALAWI',
    'MALAYSIA',
    'MALI',
    'MALTA',
    'MARTINIQUE',
    'MAURITIUS',
    'MEXICO',
    'MOLDOVIA',
    'MOROCCO',
    'NEPAL',
    'NETHERLANDS',
    'NEW ZEALAND',
    'NICARAGUA',
    'NORWAY',
    'Nigeria',
    'PANAMA',
    'PAPUA NEW GUINE',
    'PARAGUAY',
    'PERU',
    'PHILIPPINES',
    'POLAND',
    'PORTUGAL',
    'REPUBLIC OF THE CONGO',
    'ROMANIA',
    'RUSSIA',
    'RWANDA',
    'SENEGAL',
    'SEYCHELL ISLANDS',
    'SINGAPORE',
    'SLOVAKIA',
    'SLOVENIA',
    'SOUTH AFRICA',
    'SOUTH KOREA',
    'SPAIN',
    'SRI LANKA',
    'SWEDEN',
    'SWITZERLAND',
    'Serbia & Montenegro',
    'TAIWAN',
    'TANZANIA',
    'THAILAND',
    'TRINIDAD And TOBAGO',
    'TURKEY',
    'Tajikistan',
    'UGANDA',
    'UKRAINE',
    'UNITED ARAB EMIRATES',
    'UNITED KINGDOM',
    'URUGUAY',
    'USA',
    'Uzbekistan',
    'VIETNAM',
    'YUGOSLAVIA',
    'ZAMBIA',
    'ZIMBABWE'
        )

        routes = routes_air

    else:
        loading_ports = (
            'ANTWERPEN',
    'Ashdod',
    'BARCELONA',
    'BOSTON',
    'BREMERHAVEN',
    'CHICAGO',
    'GENOA',
    'HAIFA',
    'HAIFA BAY PORT',
    'HAMBURG',
    'HOUSTON - TX',
    'KOPER',
    'LE HAVRE',
    'MERSIN',
    'MUNDRA',
    'NEW YORK, NY',
    'NORFOLK, VA',
    'RICE LAKE',
    'ROTTERDAM',
    'SANTOS',
    'SAVANNAH',
    'SHANGHAI',
    'SOUTH PORT',
    'TORONTO'
        )

        loading_countries = (
            'BELGIUM',
    'BRAZIL',
    'CANADA',
    'CHINA',
    'FRANCE',
    'GERMANY',
    'INDIA',
    'ISRAEL',
    'ITALY',
    'NETHERLANDS',
    'SLOVENIA',
    'SPAIN',
    'TURKEY',
    'USA'
        )

        destination_ports = (
            'ABIDJAN',
    'ABU DHABI',
    'ACAJUTLA',
    'ALGECIRAS',
    'ALTAMIRA',
    'AMBARLI',
    'ANTALYA',
    'ANTWERPEN',
    'APAPA',
    'AQABA (EL AKABA)',
    'ARHUS',
    'ASHDOD',
    'ASUNCION',
    'AUCKLAND',
    'BALBOA',
    'BAMAKO',
    'BANGKOK',
    'BARCELONA',
    'BARRANQUILLA',
    'BEIRA',
    'BELFAST',
    'BOSTON',
    'BRISBANE',
    'BUENOS AIRES',
    'BUSAN (EX PUSAN)',
    'Bridgetown',
    'CAGLIARI',
    'CALLAO',
    'CAPE TOWN',
    'CARTAGENA',
    'CASABLANCA',
    'CAT LAI',
    'CATANIA',
    'CAUCEDO',
    'CHARLESTON',
    'CHARLESTON, SC',
    'CHENNAI (EX MADRAS)',
    'CHICAGO',
    'CHICO, CA',
    'COCHIN',
    'COLOMBO',
    'COLON',
    'COLUMBUS - OH',
    'CONAKRY',
    'CONSTANTA',
    'COTONOU',
    'DA CHAN BAY',
    'DAKAR',
    'DAMIETTA',
    'DAR ES SALAAM',
    'DAVAO',
    'DJIBOUTI',
    'DOUALA',
    'DUBLIN',
    'DURBAN',
    'DURRES',
    'EL ISKANDARIYA (= ALEXANDRIA)',
    'EL SALVADOR',
    'EVYAPAN',
    'FAMAGUSTA',
    'FELIXSTOWE',
    'FORT-DE-FRANCE',
    'FOS-SUR-MER',
    'FREDERICIA',
    'FUKUOKA',
    'GDANSK',
    'GEMLIK',
    'GENERAL SANTOS',
    'GENOA',
    'GEORGETOWN',
    'GIJON',
    'GOTEBORG',
    'GUANGZHOU',
    'GUATEMALA CITY',
    'GUAYAQUIL',
    'HAIFA',
    'HAIPHONG',
    'HAKATA',
    'HALIFAX',
    'HAMBURG',
    'HAZIRA',
    'HELSINGBORG',
    'HELSINKI (HELSINGFORS)',
    'HO CHI MINH CITY',
    'HONG KONG',
    'HOUSTON - TX',
    'HUANGPU',
    'INCHON',
    'INVERCARGILL',
    'ISTANBUL',
    'ITAJAI',
    'IZMIR',
    'IZMIR (SMYRNA)',
    'JAKARTA',
    'JEBEL ALI',
    'KAOHSIUNG',
    'KEELUNG (CHILUNG)',
    'KINGSTON',
    'KLAIPEDA',
    'KOBENHAVN',
    'KOCHI',
    'KOPER',
    'KOTKA',
    'LA GUAIRA',
    'LAEM CHABANG',
    'LARVIK',
    'LAVRION (LAURIUM)',
    'LE HAVRE',
    'LIBREVILLE',
    'LIMASSOL',
    'LISBON',
    'LIVERPOOL',
    'LOME',
    'LONDON GATEWAY',
    'LONG BEACH',
    'LOS ANGELES - LA',
    'MALTA (VALETTA)',
    'MANILA NORTH HARBOUR',
    'MARIEL',
    'MATADI',
    'MELBOURNE',
    'MERSIN',
    'MIAMI, FL',
    'MOMBASA',
    'MONTEVIDEO',
    'MONTREAL',
    'MUNDRA',
    'Manila',
    'Mexico City',
    'NAGOYA',
    'NANSHA',
    'NAVEGANTES',
    'NEW YORK, NY',
    'NHAVA SHEVA (JAWAHARLAL NEHRU)',
    'NILES',
    'NORFOLK',
    'NORFOLK, VA',
    'NOVOROSSIYSK',
    'O HARE APT/CHICAGO',
    'OAKLAND-CA',
    'OSAKA',
    'OSLO',
    'PALERMO',
    'PANAMA CITY',
    'PECEM',
    'PHNOM PENH',
    'PIPAVAV (VICTOR) PORT',
    'PIRAEUS',
    'POINT LISAS',
    'POINTE DES GALETS',
    'POINTE NOIRE',
    'PORT ELIZABETH',
    'PORT LOUIS',
    'POTI',
    'PUERTO BARRIOS',
    'PUERTO CORTES',
    'PUERTO QUETZAL',
    'Port of Spain',
    'Puerto Angamos',
    'QINZHOU',
    'RAUMO (RAUMA)',
    'RAVENNA',
    'RIO DE JANEIRO',
    'ROTTERDAM',
    'SALERNO',
    'SAN ANTONIO - CHILE',
    'SAN JUAN',
    'SAN SALVADOR',
    'SANTO TOMAS DE CASTILLA',
    'SANTOS',
    'SAVANNA',
    'SAVANNAH',
    'SEATTLE',
    'SHANGHAI',
    'SIHANOUKVILLE (KOMPONG SAOM)',
    'SINES',
    'SINGAPORE',
    'SURABAYA',
    'SYDNEY',
    'Santo Domingo',
    'TAICHUNG',
    'TAMATAVE (TOAMASINA)',
    'TAURANGA',
    'TEMA',
    'THESSALONIKI',
    'TIANJIN',
    'TIANJINXINGANG',
    'TINCAN/LAGOS',
    'TOKYO',
    'TOMAKOMAI',
    'TORONTO',
    'TRIESTE',
    'Tallinn',
    'VALENCIA',
    'VALPARAISO',
    'VANCOUVER',
    'VARNA',
    'VENEZIA',
    'VERACRUZ',
    'XIAMEN',
    'YANTIAN',
    'YARIMCA',
    'YOKOHAMA'
        )

        destination_countries = (
            'ALBANIA',
    'ARGENTINA',
    'AUSTRALIA',
    'BARBADOS',
    'BELGIUM',
    'BENIN',
    'BRAZIL',
    'BULGARIA',
    'CAMBODIA',
    'CAMEROON',
    'CANADA',
    'CHILE',
    'CHINA',
    'COLOMBIA',
    'CONGO (brazaville)',
    'CUBA',
    'CYPRUS',
    'DENMARK',
    'DJIBOUTI',
    'DOMINICAN REPUBLIC',
    'ECUADOR',
    'EGYPT',
    'EL SALVADOR',
    'ESTONIA',
    'FINLAND',
    'FRANCE',
    'GABON',
    'GEORGIA',
    'GERMANY',
    'GHANA',
    'GREECE',
    'GUATEMALA',
    'GUYANA',
    'HONDURAS',
    'HONG KONG',
    'INDIA',
    'INDONESIA',
    'IRELAND',
    'ISRAEL',
    'ITALY',
    'IVORY COAST',
    'JAMAICA',
    'JAPAN',
    'JORDAN',
    'KENYA',
    'LITHUANIA',
    'MADAGASCAR',
    'MALI',
    'MALTA',
    'MARTINIQUE',
    'MAURITIUS',
    'MEXICO',
    'MOROCCO',
    'MOZAMBIQUE',
    'NETHERLANDS',
    'NEW ZEALAND',
    'NORWAY',
    'Nigeria',
    'PANAMA',
    'PARAGUAY',
    'PERU',
    'PHILIPPINES',
    'POLAND',
    'PORTUGAL',
    'PUERTO RICO',
    'REPUBLIC OF THE CONGO',
    'REUNION',
    'ROMANIA',
    'RUSSIA',
    'SENEGAL',
    'SINGAPORE',
    'SLOVENIA',
    'SOUTH AFRICA',
    'SOUTH KOREA',
    'SPAIN',
    'SRI LANKA',
    'SWEDEN',
    'TAIWAN',
    'TANZANIA',
    'THAILAND',
    'TOGO',
    'TRINIDAD And TOBAGO',
    'TURKEY',
    'UNITED ARAB EMIRATES',
    'UNITED KINGDOM',
    'URUGUAY',
    'USA',
    'VENEZUELA',
    'VIETNAM'
        )

        routes = routes_ocean


    # Dropdown for choosing the loading country
    loading_country = st.selectbox(
        'Select the loading country:',
        loading_countries
    )

    # Dropdown for choosing the loading port
    loading_port = st.selectbox(
        'Select the loading port:',
        loading_ports
    )

    # Dropdown for choosing the destination country
    destination_country = st.selectbox(
        'Select the destination country:',
        destination_countries
    )

    # Dropdown for choosing the destination port
    destination_port = st.selectbox(
        'Select the destination port:',
        destination_ports
    )


    # Multiple choice for the number of legs
    legs = st.multiselect(
        'Select the number of legs:',
        [1, 2, 3, 4, 5, 6, 7]
    )
    ##
    # Display routes based on the selected number of legs
    routes_by_leg = {}
    for leg in legs:
        routes_by_leg[leg] = routes.get(str(leg), [])

    # Combine all routes into a single list for multiselect
    all_routes = []
    for leg_routes in routes_by_leg.values():
        all_routes.extend(leg_routes)
    ##

    # # Display routes based on the selected number of legs
    # selected_routes = []
    # for leg in legs:
    #     selected_routes.extend(routes.get(str(leg), []))

    # Multiselect for routes
    selected_routes = st.multiselect(
        'Select the routes:',
        all_routes
    )

    # Save list of results per user
    results = []


    # Create a submit button
    if st.button('Submit'):
        if transport_type == 'Air':
            if selected_routes and legs:
                for leg in legs:
                    valid_routes = [route for route in selected_routes 
                                    if route in routes_by_leg[leg]]
                    for route in valid_routes:
                        st.write(f"Selected route: {route} with {leg} legs")
                        try:
                            result_table = predict_top_companies(
                                loading_port, 
                                loading_country, 
                                destination_port, 
                                destination_country, 
                                route,
                                leg
                            )

                            st.write(f"Predicted Top 3 {transport_type} Shipping Companies are: ")
                            st.dataframe(result_table)

                            # Collect the result table in results list
                            results.append(result_table)

                            #Show feature importance using RandomForest and SHAP
                            for company_name in result_table['Predicted Shipping Companies']:
                                st.write("Feature Importance for: ", company_name)
                                process_and_visualize(company_name)     

                        except Exception as e:
                            st.error(e)
            else:
                st.write("Please select at least one leg and one route.")

        elif transport_type == 'Ocean':
            if selected_routes and legs:
                for leg in legs:
                    valid_routes = [route for route in selected_routes 
                                    if route in routes_by_leg[leg]]
                    for route in valid_routes:
                        st.write(f"Selected route: {route} with {leg} legs")
                        try:
                            result_table = predict_top_companies(
                                loading_port, 
                                loading_country, 
                                destination_port, 
                                destination_country, 
                                route,
                                leg
                            )

                            st.write(f"Predicted Top 3 {transport_type} Shipping Companies are: ")
                            st.dataframe(result_table)

                            # Collect result tables 
                            results.append(result_table)

                            #Show feature importance using RandomForest and SHAP
                            for company_name in result_table['Predicted Shipping Companies']:
                                st.write("Feature Importance for: ", company_name)
                                process_and_visualize(company_name)                         

                        except Exception as e:
                            st.error(e)
            else:
                st.write("Please select at least one leg and one route.")

      # Save and download results at the end
    if results:
        all_results_df = pd.concat(results, ignore_index=True)
        
        # Save the result
        auth.save_result(username, all_results_df)
        
        # Provide download option
        csv = all_results_df.to_csv(index=False).encode()
        st.download_button(
            label="Download all results as CSV",
            data=csv,
            file_name='all_search_results.csv',
            mime='text/csv',
        )
    

if __name__ == "__main__":
    main()
