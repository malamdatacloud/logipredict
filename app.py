import shap
import matplotlib
matplotlib.use("agg")
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
from st_aggrid import AgGrid, GridOptionsBuilder
shap.initjs()

from air.air import *
from ocean.ocean import *

# Set Title for the APP
st.title('LogiPredict: Smart Shipping Solutions')

# Dropdown for choosing between Air and Ocean
transport_type = st.selectbox(
    'Choose your shipping transport method: ',
    ('Air', 'Ocean')
)

# Define dynamic lists for loading and destination options based on the shipping method
if transport_type == 'Air':

    loading_ports = (
        'Tel-Aviv', 'MANCHESTER', 'SHANGHAI', 'AMSTERDAM', 'FRANKFURT',
       'Dallas,tx', 'Houston,tx', 'TORONTO', 'Stuttgart', 'Miami,fl',
       'CALCUTTA', 'CHICAGO', 'PUDONG', 'ATLANTA', 'INCHEON', 'SHENZHEN',
       'DALIAN', 'HONG-KONG', 'GOTHENBURG', 'JINAN', 'BARCELONA',
       'HOUSTON', 'MONTREAL', 'TAIPEI', 'MILANO', 'SEATTLE', 'PORTLAND',
       'JOHN F.KENNEDY', 'CHARLES DE GAULLE', 'CHENNAI', 'LOS ANGELES',
       'LONDON-HEATHROW', 'DALLAS', 'SINGAPOR', 'Tokyo Narita', 'PEKING',
       'NEWARK', 'LJUBLJANA', 'LIEGE', 'ZURICH', 'ZHENGZHOU', 'ATHENS',
       'COPENHAGEN', 'WARSAW', 'HAMBURG', 'HELSINKI', 'BANGALOR',
       'JOHANSBURG', 'Muenchen', 'MADRID', 'PORTO', 'PRAGUE', 'HYDERABAD',
       'BOMBAY', 'BILLUND', 'HANOI', 'ISTANBUL', 'Cologne', 'SOFIA',
       'DUBAI', 'DUBLIN', 'BOSTON', 'VIENNA', 'BANGKOK', 'IZMIR',
       'Berlin', 'OSAKA', 'LIMA', 'ZHEJIANG', 'XIAN', 'MANILA',
       'GUANGZHOU', 'LARNACA', 'BRUSSLES', 'MAASTRICHT', 'CHANGSHA',
       'BUCHAREST', 'BUDAPEST', 'MIAMI', 'DRESDEN', 'VANCOUVER', 'BILBAO',
       'WUHAN', 'Dusseldorf', 'GUANGDONG', 'MEXICO CITY', 'DONGGUAN',
       'LISBON', 'PERTH', 'MELBOURNE', 'MALMOE', 'cebu', 'STOCKHOLM',
       'BASEL', 'CHENGDU', 'RIGA', 'SYDNEY', 'SEGRATE', 'SAN JOSE',
       'NAGOYA', 'LINZ', 'THESSALONIKI', 'NAIROBI', 'LONDON',
       'WASHINGTON', 'DELHI', 'AHMEDABAD', 'SEOUL', 'BUENOS AIRES',
       'SANTIAGO', 'TOKYO', 'AHMEDBAD', 'BEIJING', 'QUITO',
       'JUAN SANTAMARIA', 'BOGOTA', 'BRATISLAVA', 'Guarulhos', 'PARIS',
       'BRISBANE', 'XIAMEN', 'FLORENCE', 'OSLO', 'CAPE TOWN', 'SUN JUAN',
       'BELGRADE', 'LYONS', 'VIRACOPOS', 'EINDHOVEN', 'HO CHI MINH CITY',
       'QINGDAO', 'KIEV', 'Guadalajara', 'FUZHOU', 'JAKARTA', 'ANKARA',
       'COLOMBO', 'KATOWICE', 'MOSCOW', 'LODZ', 'AUCKLAND', 'HANOVER',
       'PHENIX', 'CINCINNATI', 'ORLANDO', 'XIANYANG', 'LAS VEGAS'
    )

    loading_countries = (
        'ISRAEL', 'UNITED KINGDOM', 'CHINA', 'NETHERLANDS', 'GERMANY',
       'USA', 'CANADA', 'INDIA', 'SOUTH KOREA', 'HONG KONG', 'SWEDEN',
       'SPAIN', 'TAIWAN', 'ITALY', 'FRANCE', 'SINGAPORE', 'JAPAN',
       'SLOVENIA', 'BELGIUM', 'SWITZERLAND', 'GREECE', 'DENMARK',
       'POLAND', 'FINLAND', 'SOUTH AFRICA', 'PORTUGAL', 'CZECH REPUBLIC',
       'VIETNAM', 'TURKEY', 'BULGARIA', 'UNITED ARAB EMIRATES', 'IRELAND',
       'AUSTRIA', 'THAILAND', 'PERU', 'PHILIPPINES', 'CYPRUS', 'ROMANIA',
       'HUNGARY', 'MEXICO', 'AUSTRALIA', 'LATVIA', 'KENYA',
       'KOREA - south', 'ARGENTINA', 'CHILE', 'ECUADOR', 'PUERTO RICO',
       'COLOMBIA', 'SLOVAKIA', 'BRAZIL', 'NORWAY', 'SURINAM', 'UKRAINE',
       'INDONESIA', 'SRI LANKA', 'RUSSIA', 'NEW ZEALAND'
    )

    destination_ports = (
        'Tokyo - Narita Airport', 'Frankfurt', 'Beijing', 'Memphis, tn',
       'Maastricht', 'Bombay - Mumbai', 'Newark, NJ', 'Guadalajara',
       'Dallas,tx', 'Jakarta - Sukarno Hatta', 'TORONTO', 'Lima',
       'NEW YORK, NY', 'Sao Paulo-Guarulhos', 'Houston,tx', 'Havanna',
       'London-Heathrow', 'Berlin', 'BUENOS AIRES',
       'Paris-Charles de Gaulle', 'San Francisco,ca', 'Chicago Ohare,il',
       'DUBLIN', 'Milan', 'Milan - Malpensa', 'HONG KONG', 'LIBREVILLE',
       'Madrid', 'SINGAPORE', 'Seoul-Incheon Airport', 'Bangalore',
       'MUNICH', 'Mexico City', 'Amsterdam', 'Johannesburg', 'SHANGHAI',
       'Santo Domingo', 'Ho Chi Minh City (Saigon)', 'Chennai ( Madras )',
       'Liege', 'BANGKOK', 'OSLO', 'Moscow - Domodedovo',
       'GUATEMALA CITY', 'Pittsburgh,pa', 'Ahmedabad', 'CAPE TOWN',
       'Skopje', 'Miami,fl', 'Bogota', 'Portland,or', 'Curitiba,pr',
       'HAMBURG', 'Prague', 'Santiago De Chile', 'MELBOURNE', 'ISTANBUL',
       'Geneva', 'Dresden', 'Los Angeles,ca', 'Atlanta,ga', 'Lyon',
       'Richmond,va', 'Salt Lake City,ut', 'Zuerich (Zurich)',
       'Moscow-Vnukovo', 'Taipei-Chiang Kaisek', 'Kilimanjaro',
       'Warschau', 'Bridgetown', 'Osaka - Kansai Airport', 'Mcallen,tx',
       'Athens', 'Entebbe', 'Nairobi', 'San Juan,pr', 'Montreal-Dorval',
       'Brussels', 'Stockholm', 'Antigua', 'Lagos',
       'Cologne (Koeln) / Bonn', 'Zagreb', 'Grand Rapids,mi', 'Delhi',
       'Belgrad(Beograd)', 'Larnaca', 'BRISBANE', 'Seattle, WA', 'Dubai',
       'Tashkent', 'Sofia', 'Moscow-Sheremetyevo 1', 'SYDNEY', 'San Jose',
       'Hyderabad', 'Lusaka', 'Detroit,mi', 'Shenzhen', 'LISBON', 'OSAKA',
       'Duesseldorf', 'Adana', 'Quito', 'Port of Spain', 'BARCELONA',
       'Sao Paulo-Viracopos', 'DAKAR', 'Manila', 'SAN SALVADOR',
       'Alma Ata', 'Budapest', 'San Pedro Sula', 'Addis Abeba',
       'Gaborone', 'Abuja', 'AUCKLAND', 'Dalian', 'NAGOYA', 'Minsk',
       'Ljubljana', 'DIASS', 'Dar Es Salam (Daressalam)', 'Kuala Lumpur',
       'Manchester', 'Riga', 'Bishkek', 'Perth', 'Hanoi', 'COLOMBO',
       'Mauritius', 'Helsinki', 'Bologna', 'MONTEVIDEO',
       'Guangzhou (kanton)', 'Washington, DC', 'Chengdu', 'VALENCIA',
       'Boston,ma', 'Tbilisi', "St' Louis, MO", 'Mahe', 'PANAMA CITY',
       'GUAYAQUIL', 'COCHIN', 'Roterdam', 'Casablanca-Mohamed', 'Oporto',
       'Rome-Leonardo Da Vinci/Fuimici', 'Luqa (Valletta)',
       'Bukarest-Otopeni', 'Accra', 'Dushanbe (Duschanbe)',
       'Fort De France', 'ABU DHABI', 'Copenhagen', 'Lilongwe', 'Yerevan',
       'Wuhan', 'MONTREAL', 'El Paso,tx', 'Hannover', 'San Diego,ca',
       'Managua', 'Columbus,oh', 'Kishinev', 'ANTALYA', 'Charlotte,nc',
       'Stuttgart', 'Tampa, FL', 'Phoenix,az', 'Orlando,fl', 'Mombassa',
       'Cleveland Hopkins Internationa', 'East Midlands',
       'Jakarta - Kemayoran', 'Ankara-Esenboga', 'Teneriff South',
       'Malmoe-Sturup', 'Bratislava', 'Cairo', 'KINGSTON', 'Ercan',
       'Vilnius', 'Port Moresby', 'Conarkry', 'Denver,co', 'Harare',
       'Kigali', 'VIENNA', 'CASABLANCA', 'Louisville,ky', 'Pristina',
       'SURABAYA', 'Ougadougou', 'Philadelphia,pa', 'Austin,tx', 'BAMAKO',
       'Calgary/Banff', 'GENOA', 'Cincinnati,oh', 'ABIDJAN', 'Eindhoven',
       'Ataturk', 'Tallinn', 'Shannon (Limerick)', 'Boise,id', 'Winnipeg',
       'ODESSA', 'Volgograd', 'Kiev-Borispol', 'Minneapolis,mn', 'Malaga',
       'Luxemburg', 'Krasnodar', 'GOTEBORG', 'Antananarivo (Tanannarive)',
       'Kinshasa', 'Roberts International Airport', 'Qingdao', 'Billund',
       'DURBAN', 'Tirana', 'Baku', 'Saloniki', 'Kathmandu',
       'Buenos Aires-ezeza/Ministro P', 'Berlin-Tegel', 'Huntsville, AL',
       'Las Vegas,nv', 'Rzeszow', 'Mendoza', 'St. Petersburg (Leningrad)',
       'ASUNCION', 'Tel-Aviv'
    )

    destination_countries = (
        'JAPAN', 'GERMANY', 'CHINA', 'USA', 'NETHERLANDS', 'INDIA',
       'MEXICO', 'INDONESIA', 'CANADA', 'PERU', 'BRAZIL', 'CUBA',
       'UNITED KINGDOM', 'ARGENTINA', 'FRANCE', 'IRELAND', 'ITALY',
       'HONG KONG', 'GABON', 'SPAIN', 'SINGAPORE', 'SOUTH KOREA',
       'SOUTH AFRICA', 'DOMINICAN REPUBLIC', 'VIETNAM', 'BELGIUM',
       'THAILAND', 'NORWAY', 'RUSSIA', 'GUATEMALA', 'MACEDONIA',
       'COLOMBIA', 'CZECH REPUBLIC', 'CHILE', 'AUSTRALIA', 'TURKEY',
       'SWITZERLAND', 'TAIWAN', 'TANZANIA', 'POLAND', 'BARBADOS',
       'GREECE', 'UGANDA', 'KENYA', 'SWEDEN', 'ANTIGUA And BARBUDA',
       'Nigeria', 'CROATIA', 'Serbia & Montenegro', 'CYPRUS',
       'UNITED ARAB EMIRATES', 'Uzbekistan', 'BULGARIA', 'COSTA RICA',
       'ZAMBIA', 'PORTUGAL', 'ECUADOR', 'TRINIDAD And TOBAGO', 'SENEGAL',
       'PHILIPPINES', 'EL SALVADOR', 'KAZAKHSTAN', 'HUNGARY', 'HONDURAS',
       'ETHIOPIA', 'BOTSWANA', 'NEW ZEALAND', 'BELARUS', 'SLOVENIA',
       'MALAYSIA', 'LATVIA', 'KYRGUSTAN', 'SRI LANKA', 'MAURITIUS',
       'FINLAND', 'URUGUAY', 'GEORGIA', 'SEYCHELL ISLANDS', 'PANAMA',
       'MOROCCO', 'MALTA', 'ROMANIA', 'GHANA', 'Tajikistan', 'MARTINIQUE',
       'DENMARK', 'MALAWI', 'ARMENIA', 'NICARAGUA', 'MOLDOVIA',
       'SLOVAKIA', 'EGYPT', 'JAMAICA', 'LITHUANIA', 'PAPUA NEW GUINE',
       'GUINEA BISSAU', 'ZIMBABWE', 'RWANDA', 'AUSTRIA', 'YUGOSLAVIA',
       'BURKINA FASO', 'MALI', 'IVORY COAST', 'KOREA - south', 'ESTONIA',
       'UKRAINE', 'LUXEMBOURG', 'MADAGASCAR', 'REPUBLIC OF THE CONGO',
       'LIBERIA', 'ALBANIA', 'AZARBEIJAN', 'NEPAL', 'PARAGUAY', 'ISRAEL'
    )

else:
    loading_ports = (
        'HAIFA', 'Ashdod', 'HAIFA BAY PORT', 'NEW YORK, NY', 'SANTOS',
       'MERSIN', 'ROTTERDAM', 'SAVANNAH', 'CHICAGO', 'RICE LAKE',
       'NORFOLK, VA', 'HAMBURG', 'KOPER', 'SOUTH PORT', 'BARCELONA',
       'HOUSTON - TX', 'MUNDRA', 'GENOA', 'TORONTO', 'SHANGHAI',
       'LE HAVRE', 'ANTWERPEN', 'BOSTON', 'BREMERHAVEN', 'AUCKLAND',
       'HAIPHONG', 'QINGDAO', 'KOBE', 'CARTAGENA', 'LIVERPOOL',
       'VALENCIA', 'NANJING', 'NEW YORK', 'NINGBO', 'LAEM CHABANG',
       'BUSAN (EX PUSAN)', 'FELIXSTOWE', 'DALIAN', 'SHEKOU',
       'DA CHAN BAY', 'GUANGZHOU', 'SHENZHEN', 'NORFOLK', 'HONG KONG',
       'ISTANBUL', 'KEELUNG', 'YOKOHAMA', 'XIAMEN', 'SINGAPORE',
       'RAVENNA', 'TARRAGONA', 'IZMIR', 'THESSALONIKI', 'NAVEGANTES',
       'ALIAGA', 'IZMIT', 'HALIFAX', 'HO CHI MINH CITY', 'TIANJIN',
       'SALERNO', 'BANGKOK', 'NHAVA SHEVA (JAWAHARLAL NEHRU)', 'XINGANG',
       'KAOHSIUNG', 'MARSEILLE', 'GEBZE', 'GDANSK', 'NAGOYA', 'PIRAEUS',
       'VENEZIA', 'WUHAN', 'DUBLIN', 'VALPARAISO', 'LEIXOES',
       'JACKSONVILLE', 'MONTREAL', 'AARHUS', 'PORT ELIZABETH',
       'FOS SUR MER', 'FLIXTON', 'JEBEL ALI', 'HOUSTON', 'AMBARLI',
       'CHENNAI (EX MADRAS)', 'SURABAYA', 'LIMASSOL', 'VERACRUZ',
       'OAKLAND', 'MANHATTAN', 'TAOYUAN', 'TOKYO', 'GOTEBORG',
       'CASTELLON DE LA PLANA', 'CONSTANTA', 'GDYNIA',
       'KOLKATA (EX CALCUTTA)', 'AHMEDABAD', 'AMBERLEY', 'HAARBY',
       'MANILA', 'LONG BEACH', 'LONDON GATEWAY', 'NANSHA', 'MARIN',
       'HUANGPU', 'JAKARTA', 'MONFALCONE', 'YARIMCA', 'ESBJERG',
       'ISKENDERUN', 'EVYAP', 'DERINCE', 'GEMLIK',
       'HELSINKI (HELSINGFORS)', 'BARRANQUILLA', 'TAICHUNG', 'YANTIAN',
       'RAMSTEIN-MIESENBACH', 'FUZHOU', 'SAN ANTONIO', 'SYDNEY',
       'MINNEAPOLIS', 'KALMAR', 'BORDING', 'VARNA', 'LISBOA', 'SAVANNA',
       'ULEABORG (OULU)', 'LOS ANGELES', 'MELBOURNE', 'LIANYUNGANG',
       'IZMIR (SMYRNA)', 'ODESSA', 'DELHI', 'TUTICORIN', 'BUENOS AIRES',
       'LIVORNO', 'TAIZHOU', 'NANTONG', 'JIUJIANG', 'KARACHI', 'OSAKA',
       'YANGZHOU', 'NAPOLI', 'SEMARANG', 'BUCURESTI', 'EDMONTON',
       'NEWARK', 'TALLINN', 'FREMANTLE', 'FOS-SUR-MER', 'LYTTELTON',
       'CASTELLON DE RUGAT', 'ITAJAI', 'FOSHAN', 'BENGBU', 'BALTIMORE',
       'ZHUAHAI', 'BRUXELLES (BRUSSEL)', 'ANTALYA', 'KAUNAS', 'GUAYAQUIL',
       'SEATTLE', 'HELSINGBORG', 'PIPAVAV (VICTOR) PORT', 'SHUNDE',
       'TAURANGA', 'ALTAMIRA', 'KOTKA', 'ALGECIRAS', 'TRIESTE',
       'NOVOROSSIYSK', 'NEW ORLEANS', 'WARSASH', 'SHANTOU', 'BUDAPEST',
       'RIGA', 'SOUTHAMPTON', 'TIANJINXINGANG'
    )

    loading_countries = (
        'ISRAEL', 'USA', 'BRAZIL', 'TURKEY', 'NETHERLANDS', 'GERMANY',
       'SLOVENIA', 'SPAIN', 'INDIA', 'ITALY', 'CANADA', 'CHINA', 'FRANCE',
       'BELGIUM', 'NEW ZEALAND', 'VIETNAM', 'JAPAN', 'COLOMBIA',
       'UNITED KINGDOM', 'THAILAND', 'SOUTH KOREA', 'HONG KONG', 'TAIWAN',
       'SINGAPORE', 'GREECE', 'POLAND', 'IRELAND', 'CHILE', 'PORTUGAL',
       'DENMARK', 'SOUTH AFRICA', 'UNITED ARAB EMIRATES', 'INDONESIA',
       'CYPRUS', 'MEXICO', 'SWEDEN', 'ROMANIA', 'AUSTRALIA',
       'PHILIPPINES', 'FINLAND', 'BULGARIA', 'KOREA - south', 'UKRAINE',
       'ARGENTINA', 'PAKISTAN', 'ESTONIA', 'VENEZUELA', 'LITHUANIA',
       'ECUADOR', 'RUSSIA', 'HUNGARY', 'LATVIA'
    )

    destination_ports = (
        'VALENCIA', 'NEW YORK, NY', 'TORONTO', 'KOPER', 'BARCELONA',
       'DURBAN', 'ANTWERPEN', 'ISTANBUL', 'SHANGHAI', 'TOKYO', 'ALTAMIRA',
       'ROTTERDAM', 'HUANGPU', 'MERSIN', 'NHAVA SHEVA (JAWAHARLAL NEHRU)',
       'LONDON GATEWAY', 'BANGKOK', 'LIVERPOOL', 'CHARLESTON, SC',
       'CAPE TOWN', 'SINGAPORE', 'RAVENNA', 'CAT LAI', 'COLOMBO',
       'VERACRUZ', 'IZMIR', 'CONSTANTA', 'BUSAN (EX PUSAN)', 'LIMASSOL',
       'GEMLIK', 'FOS-SUR-MER', 'NORFOLK, VA', 'BUENOS AIRES', 'SAVANNAH',
       'TRIESTE', 'VARNA', 'SAN ANTONIO - CHILE', 'GENOA',
       'TIANJINXINGANG', 'ASHDOD', 'CARTAGENA', 'EVYAPAN', 'TIANJIN',
       'HAMBURG', 'DAKAR', 'TAICHUNG', 'SANTOS', 'OSAKA', 'CAUCEDO',
       'TOMAKOMAI', 'TINCAN/LAGOS', 'SALERNO', 'CASABLANCA', 'HALIFAX',
       'GUATEMALA CITY', 'MELBOURNE', 'HOUSTON - TX', 'DUBLIN',
       'FELIXSTOWE', 'ITAJAI', 'NOVOROSSIYSK', 'FORT-DE-FRANCE', 'KOTKA',
       'POINTE DES GALETS', 'BELFAST', 'KAOHSIUNG', 'CATANIA',
       'PUERTO BARRIOS', 'EL ISKANDARIYA (= ALEXANDRIA)', 'POTI',
       'JEBEL ALI', 'HO CHI MINH CITY', 'HONG KONG', 'KOCHI',
       'LAEM CHABANG', 'VALPARAISO', 'KEELUNG (CHILUNG)', 'POINT LISAS',
       'LOME', 'SYDNEY', 'FREDERICIA', 'BAMAKO', 'DAVAO',
       'CHENNAI (EX MADRAS)', 'SAN SALVADOR', 'VENEZIA', 'INVERCARGILL',
       'TAURANGA', 'SURABAYA', 'GOTEBORG', 'GUAYAQUIL', 'GUANGZHOU',
       'MOMBASA', 'BRISBANE', 'NAGOYA', 'BOSTON', 'LONG BEACH',
       'LE HAVRE', 'FUKUOKA', 'CHARLESTON', 'CAGLIARI', 'MIAMI, FL',
       'SINES', 'SANTO TOMAS DE CASTILLA', 'MONTEVIDEO', 'YOKOHAMA',
       'COLON', 'HELSINKI (HELSINGFORS)', 'JAKARTA', 'ABIDJAN', 'CALLAO',
       'ACAJUTLA', 'COLUMBUS - OH', 'NILES', 'PIRAEUS', 'INCHON',
       'CONAKRY', 'AUCKLAND', 'BARRANQUILLA', 'COTONOU', 'DAR ES SALAAM',
       'KOBENHAVN', 'DA CHAN BAY', 'EL SALVADOR', 'AQABA (EL AKABA)',
       'DJIBOUTI', 'TAMATAVE (TOAMASINA)', 'LOS ANGELES - LA', 'KLAIPEDA',
       'PANAMA CITY', 'OSLO', 'HAIFA', 'TEMA', 'LIBREVILLE', 'DOUALA',
       'ALGECIRAS', 'LARVIK', 'PUERTO CORTES', 'GEORGETOWN', 'NAVEGANTES',
       'MALTA (VALETTA)', 'PHNOM PENH', 'PORT LOUIS', 'Tallinn',
       'FAMAGUSTA', 'MANILA NORTH HARBOUR', 'POINTE NOIRE', 'YANTIAN',
       'HAIPHONG', 'KINGSTON', 'DURRES', 'CHICAGO', 'NANSHA',
       'Port of Spain', 'PALERMO', 'SEATTLE', 'THESSALONIKI', 'APAPA',
       'MUNDRA', 'PIPAVAV (VICTOR) PORT', 'IZMIR (SMYRNA)',
       'Puerto Angamos', 'ANTALYA', 'HELSINGBORG', 'NORFOLK', 'MONTREAL',
       'Santo Domingo', 'CHICO, CA', 'Bridgetown', 'YARIMCA', 'HAKATA',
       'PUERTO QUETZAL', 'MATADI', 'PORT ELIZABETH', 'LISBON', 'Manila',
       'ARHUS', 'QINZHOU', 'MARIEL', 'ASUNCION', 'BEIRA', 'Mexico City',
       'VANCOUVER', 'HAZIRA', 'ABU DHABI', 'XIAMEN', 'PECEM', 'BALBOA',
       'SIHANOUKVILLE (KOMPONG SAOM)', 'GENERAL SANTOS', 'COCHIN',
       'LAVRION (LAURIUM)', 'SAN JUAN', 'O HARE APT/CHICAGO', 'LA GUAIRA',
       'DAMIETTA', 'GDANSK', 'RAUMO (RAUMA)', 'OAKLAND-CA',
       'RIO DE JANEIRO', 'SAVANNA', 'HAIFA BAYPORT', 'SOUTH PORT',
       'OVERSEAS ASHDOD', 'HAIFA MIFRATZ'
    )

    destination_countries = (
        'SPAIN', 'USA', 'CANADA', 'SLOVENIA', 'SOUTH AFRICA', 'BELGIUM',
       'TURKEY', 'CHINA', 'JAPAN', 'MEXICO', 'NETHERLANDS', 'INDIA',
       'UNITED KINGDOM', 'THAILAND', 'SINGAPORE', 'ITALY', 'VIETNAM',
       'SRI LANKA', 'ROMANIA', 'SOUTH KOREA', 'CYPRUS', 'FRANCE',
       'ARGENTINA', 'BULGARIA', 'CHILE', 'ISRAEL', 'COLOMBIA', 'GERMANY',
       'SENEGAL', 'TAIWAN', 'BRAZIL', 'DOMINICAN REPUBLIC', 'Nigeria',
       'MOROCCO', 'GUATEMALA', 'AUSTRALIA', 'IRELAND', 'RUSSIA',
       'MARTINIQUE', 'FINLAND', 'REUNION', 'EGYPT', 'GEORGIA',
       'UNITED ARAB EMIRATES', 'HONG KONG', 'TRINIDAD And TOBAGO', 'TOGO',
       'DENMARK', 'MALI', 'PHILIPPINES', 'EL SALVADOR', 'NEW ZEALAND',
       'INDONESIA', 'SWEDEN', 'ECUADOR', 'KENYA', 'PORTUGAL', 'URUGUAY',
       'PANAMA', 'IVORY COAST', 'PERU', 'GREECE', 'GUYANA', 'BENIN',
       'TANZANIA', 'JORDAN', 'DJIBOUTI', 'MADAGASCAR', 'LITHUANIA',
       'NORWAY', 'GHANA', 'GABON', 'CAMEROON', 'HONDURAS', 'MALTA',
       'CAMBODIA', 'MAURITIUS', 'ESTONIA', 'CONGO (brazaville)',
       'JAMAICA', 'ALBANIA', 'BARBADOS', 'REPUBLIC OF THE CONGO', 'CUBA',
       'PARAGUAY', 'MOZAMBIQUE', 'PUERTO RICO', 'VENEZUELA', 'POLAND'
    )

################### test #####################


######################## end test #######################


# Dropdown for choosing the loading port
loading_port = st.selectbox(
    'Select the loading port:',
    loading_ports
)

# Dropdown for choosing the loading country
loading_country = st.selectbox(
    'Select the loading country:',
    loading_countries
)

# Dropdown for choosing the destination port
destination_port = st.selectbox(
    'Select the destination port:',
    destination_ports
)

# Dropdown for choosing the destination country
destination_country = st.selectbox(
    'Select the destination country:',
    destination_countries
)

# Numeric input for choosing the number of legs
legs = st.number_input(
    'Enter the number of legs:',
    min_value=0, max_value=7, value=0, step=1
)


# Create a submit button
if st.button('Submit'):
    if transport_type == 'Air':
        
        result_table, shap_figures = integrated_prediction_and_visualization_a2(
            loading_port, loading_country, destination_port, destination_country, legs
        )

        st.write(f"Predicted Top 3 {transport_type} Shipping Companies from {loading_port} - {loading_country} to {destination_port} - {destination_country} are: ")
        
        # Display the result table with AgGrid
        gb = GridOptionsBuilder.from_dataframe(result_table)
        gb.configure_selection('single', use_checkbox=False)
        grid_options = gb.build()

        grid_response = AgGrid(
            result_table,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=True,
            theme='streamlit',
            height=300,
            width='100%'
        )

        selected = grid_response['selected_rows']

        if selected:
           selected_company = selected[0]['Predicted Shipping Company']
           st.write(f"Selected company: {selected_company}") 
           process_and_visualize_a2(selected_company, df_a2, company_to_encoded_a2)
           st.pyplot()

    elif transport_type == 'Ocean':
        result_table, shap_figures = integrated_prediction_and_visualization_o2(
            loading_port, loading_country, destination_port, destination_country, legs
        )

        st.write(f"Predicted Top 3 {transport_type} Shipping Companies from {loading_port} - {loading_country} to {destination_port} - {destination_country} are: ")
        
        # Display the result table with AgGrid
        gb = GridOptionsBuilder.from_dataframe(result_table)
        gb.configure_selection('single', use_checkbox=False)
        grid_options = gb.build()

        grid_response = AgGrid(
            result_table,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=True,
            theme='streamlit',
            height=300,
            width='100%'
        )

        selected = grid_response['selected_rows']

        if selected:
            selected_company = selected[0]['Predicted Shipping Company']
            st.write(f"Selected company: {selected_company}") 
            process_and_visualize_a2(selected_company, df_a2, company_to_encoded_a2)
            st.pyplot()