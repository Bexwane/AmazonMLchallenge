# Error analysis

| kind | India | US |
|---|---|---|
| FN | 8202 | 11798 |
| FP | 2204 | 2358 |
| TP | 7840 | 12160 | 

## FN modes

| mode | India | US | total |
|---|---|---|---|
| empty_address_record | 3984 | 6938 | 10922 |
| other | 1408 | 2510 | 3918 |
| name_replaced_same_address | 642 | 1306 | 1948 |
| common_name_crowd | 418 | 754 | 1172 |
| native_script_name | 767 | 0 | 767 |
| address_rewritten_name_kept | 431 | 4 | 435 |
| alias_or_web_name_form | 180 | 183 | 363 |
| partial_name_and_address_noise | 187 | 53 | 240 |
| name_replaced_address_changed | 185 | 50 | 235 | 

## FP modes

| mode | India | US | total |
|---|---|---|---|
| other | 923 | 1250 | 2173 |
| empty_address_record | 211 | 440 | 651 |
| common_name_crowd | 204 | 302 | 506 |
| name_replaced_same_address | 165 | 250 | 415 |
| native_script_name | 358 | 0 | 358 |
| address_rewritten_name_kept | 161 | 16 | 177 |
| alias_or_web_name_form | 70 | 72 | 142 |
| partial_name_and_address_noise | 85 | 21 | 106 |
| name_replaced_address_changed | 27 | 7 | 34 | 

## FP: true owner of the wrongly merged record

| belongs | India | US |
|---|---|---|
| another S1 | 785 | 734 |
| unmatched (no S1) | 1419 | 1624 | 

- H1: record contains a word of its true owner's name that our S1 lacks: **5.5%** of 1519
- owner S1 has exactly our S1's address (same-address different entity): **17.2%**
- owner name words == our name words (indistinguishable names): 34.8%
- H3: both have a house number and the first numbers differ: **40.4%** of FP (both-have-number 78.5%)
- FP record's exact address shared by >=2 S2/S3 records: 42.2%
- FP record's address == our S1 address (normalized): 6.5%

## FN: evidence from the S1's other true records

- H2: missed record has exactly the address of a sibling true record: **9.2%** of 20000
- missed record shares its first house number with a sibling: 23.5%
- missed record's address == our S1 address (normalized): 3.1%
- p1 of misses: median 0.226, share p1 > 0.3: 42.5%

## FP examples

| country | mode | p_final | s1_name | s1_addr | r_name | r_addr | owner_name | owner_addr |
|---|---|---|---|---|---|---|---|---|
| India | other | 0.989 | FC Square Ltd | S4 E-35 Lajpat Nagar, Ghaziabad, Uttar Pradesh | FDC Square Ltd | S4 E-35 LAJPAT NAGAR, GHAZIABAD, Uttar Pradesh | - | - |
| US | other | 0.962 | Bautista's Desert Family Practice | 411 Williams Place, Fl 1, Chandler, AZ | DESERT PRACTICE FAMILY SMITH | 411 WILLIAMS PL, CHANDLER, AZ | - | - |
| US | other | 0.923 | Zephora Dbdr LLC | Reading, MA, 293 Haverhill Street | Zephori Dbdr (LLC) | 293 Haverhill St, Reading, Massachusetts | - | - |
| US | other | 0.962 | Brightyn Roofing P.C. | 5859 Gateway Boulevard, Mesa, AZ | The Brightyns Roofing P.C. | MESA FOUR PEAKS, 5859 GATEWHY BLVD, AZ | - | - |
| India | other | 0.963 | KL Ice | E-193, Riico Industrial Area, Mansarovar, Jaipur, Rajasthan | kjl ice | RJ, E-193, Riico Industrial Area, Mansarovar, Jaipur | - | - |
| India | partial_name_and_address_noise | 0.713 | Gangaksh Properties | Telhara P.O.- Telhara, Nalanda, Bihar | Gangacsh Properties | Nalanda, BR, Door No 27. Telhara P.o.- Telhara, Jehanabad | - | - |
| US | other | 0.940 | Bobine Rosales Four | 2590 Northstar Trail Lane, Unit 305, Columbus, OH | Rosales, Ross Four | 2590 NORTHSTAR TRIIL LANE, COLUMBUS, OH | - | - |
| India | other | 0.982 | Ziravi Mall Co | 55-42/2/1A S R Nagar Doctor'S Colony, Seethammadhara Visakhapatnam, Vishakhapatnam, Andhra Pradesh | Ziravia Mllo Co | Vishakhapatnam, Seethammadhara Visakhapatnam, 55-42/2/5A S R Nagar Doctor's Colony, AP | - | - |
| India | native_script_name | 0.871 | United Foundation Private Limited | Fl No-503 Sr No-73/4/1, Mehboob Enclave Lane N-B, Pune City, Pune, Maharashtra | यूनाइटेड फाउंडेशन प्राइवेट लिमिटेड | SHOP NO. 503, THANE, Maharashtra | United Foundation Private Limited | Shop No. 503, Iris Shopping Meadows, Pokhran Road No. 02, Thane, Maharashtra |
| India | native_script_name | 0.935 | Alpha Builders Private Limited | Plot No.1812A, Patel Nagar 66 Foota Road, Nr. St. Stephen School, Bahadurgarh, Jhajjar, Haryana | अल्फा Builders Limited | Plot No.1833a, Patel Nagar 66 Foota Road, Nr. St. Stephen School, Bahadurgarh, Jhajjar, हरियाणा | - | - |
| India | name_replaced_same_address | 0.821 | Aditya Om Infotech Private Limited | Chetana Housing Colony, Alwal, Plot No-247, Secunderabad, Hyderabad, Telangana | Yumacalo | Hyderabad, తెలంగాణ, Plot No-247, Chetana Housing Colony, Alwal, Secunderabad | Diamond India (India) Pvt. Ltd. | Plot No-247, Chetana Housing Colony, Alwal, Secunderabad, Hyderabad, Telangana |
| US | other | 0.973 | Piedmont Trust III | WI, N16040 State Highway 13, Park Falls | Piedmont Trust III Highland | N16049 State Highway 13, Park Falls, Wisconsin | - | - |
| US | other | 0.658 | Prime Digital Networks | 45753 Long Way, Maricopa, AZ | THE  PRIME DIGITAL NETWORKS PARTNERS - 4066782352 | 45756 LONG WAY, MARICOPA, AZ | - | - |
| US | other | 0.789 | Watson Creative Salon Clinic | W167N8923 Grand Avenue, Unit Apartment 419, Village Of Menomonee Falls, WI | inc watson creative salon clinic | W167N8936D GRAND AVE, MENOMONEE FALLS, WI | - | - |
| India | partial_name_and_address_noise | 0.986 | Ravika Sons Limited | Kuzhikandathil House, Kalluvayal Post Chulliyode Iritty, Kannur, Kannur, Kerala | Raviki  Sbs Limited | Kuzhikandathil House, Kannur, KL | - | - |
| US | name_replaced_same_address | 0.785 | Dental Care | 355 Park Drive, Unit Suite 131, Irondequoit, NY | Cnehe Group | # Suite 131, New York, Irondequoit, 355 Park Dr | Cenet Group | NY, Unit Suite 131, Irondequoit, 355 Park Drive |
| India | common_name_crowd | 0.944 | Reliance & Brothers Private Limited | #10A, 7Th Cross, Eco Nest Building, Vagdevi Layout, Munnekolalu, Bangalore, Karnataka | RELIANCE & BROTHERS LIMITED | #23A, 7Th Cross, Eco Nest Building, Vagdevi Layout, Bangalore, Munnekolalu, KA | - | - |
| US | name_replaced_same_address | 0.991 | Sterling | 502 Mooreland Avenue, Harrodsburg, KY | Zephnovi | 502 MOORELAND AVE, HARRODSBURG, KY | Better Better Iris | KY, 502 Mooreland Avenue, Harrodsburg |
| India | alias_or_web_name_form | 0.999 | Bhubaneswar Helping Private Limited | 362/3850C, Right Side Of The 1St Floor At Gautam Nagar, Bhubaneswar, Khordha, Orissa | bhubaneswarrealities.com | 362/3850C, Right Side Of The First Floor, Gautam Nagar, Bhubaneswar, Khordha, OD | Bhubaneswar Realities | Plot No. 362/3850C, Right Side Of The First Floor, Gautam Nagar, Bhubaneswar, Khordha, Orissa |
| India | common_name_crowd | 0.948 | Innovative Infrastructure Ltd | Mo Sadat Near Imambara, Kotwali Dehatbijnor, Nagina, Bijnor, Bijnore, Uttar Pradesh | Innovative Ínfrastructure Llp | H.no ##74 Mo Sadat Near Imambara, Bijnor, Bijnore, उत्तर प्रदेश | - | - |
| US | name_replaced_same_address | 0.981 | Backus Alternative, LLC | 127 Tryon Street, Unit 305, Charlotte, NC | Korzeta | 00127 TRYON STREET, CHARLOTTE, NC | - | - |
| India | empty_address_record | 0.712 | Dnc Wealth Pvt Ltd | 279/1 Adarsh Nagar, New Railway Road, Arjun Nagar, Gurgaon, Haryana | Dnc Weomlh Pvt Ltd |  | Dnc Wealth Pvt Ltd | 3/437 Awas Vikas, Hanspuram, Naubasta, Kasigaon, Kanpur Nagar, Uttar Pradesh |
| India | address_rewritten_name_kept | 0.999 | Star Properties Limited | 32, Swastika, Industrial Hub, Balrol, Dascroi, Ahmedabad, Gujarat | Star Properties | 32, Ahmedabad, GJ | Star Properties Private Limited | 32, Usmanpura Village, Nr. Saiprasad Flats, Ashram Road, Ahmedabad, Gujarat |
| India | other | 0.663 | Genuine International School | C-27/274-A-3-1-A, First Floor Dr. Jai Singh Building, Maldahiya, Varanasi, Uttar Pradesh | Genuine Ínternational School LLP | C-27/274-a-3-22-a, First Floor Dr. Jai Singh Building, Varanasi, Maldahiya, उत्तर प्रदेश | - | - |
| India | other | 0.992 | (india) Tanya Solutions Private Limited | 40/8080Sha Complex 3Rd Floor Mullassery Canal Road, Ernakulam, Kerala | Private (india) Tanya Services Limited | No 40/8101Sha Complex 3Rd Floor Mullassery Canal Road, Ernakulam, KL | - | - |
| India | native_script_name | 0.904 | Southern Media Private Limited | Schoolparamb House, Ernad, Kondotty, Ernad, Malappuram, Malapuram, Kerala | സതേൺ ഇൻവെസ്റ്റ്മെന്റ് പ്രൈവറ്റ് ലിമിറ്റഡ് | NO 46 SCHOOLPARAMB HOSUE, ERNAD, KONDOTTY, ERNAD, MALAPPURAM, MALAPURAM, കേരളം | - | - |
| US | other | 0.966 | Gannon, Vinny C., CPA | 1212 2890, Nibley, UT | *** 6annon, Vinny C., CA | 1212 2890, Loagn, Utah | - | - |
| India | other | 0.954 | Meenva India Private Limited | Kolkata, Everest, 46C, Chowringhee Rd P, S, Park Street.P.S. Shakespeare Sarani, Howrah, West Bengal, 5F | Meenix India Private Limited | 5F, Everest, 46C, Chowringhee Rd P, S, Park Street.p.s. Shakespeare Sarani, Kolkata, Howrah, WB | - | - |
| India | native_script_name | 0.907 | Royal Projects Pvt Ltd | Unit 102, Siddha Weston, 9, Weston Street, Kolkata, Kolkata, Howrah, West Bengal | রয়্যাল পাওয়ার প্রাইভেট লিমিটেড | SIDDHA WESTON, KOLKATA, HOWRAH, West Bengal | Royal Power Private Limited | Siddha Weston, 9, Weston Strreet, 3Rd Floor, Suite No. 321, Kolkata, Kolkata, Howrah, West Bengal |
| US | other | 0.821 | Annamarie Dailey Nexpoint Care | 150-17 Tahoe Street, Ozone Park, NY | Annamarie Dailey Nexpoint CARE CORP | 150-18b Tahoe Street, <NULL>, Ozone PARK, New York | - | - |
| India | other | 0.929 | DA Innovations Private Limited | Flat No.-40H, Pocket-A-2, Mayur Vihar, Phase 3, East Delhi, Delhi | Limited DA Innovations | Flat No.-45h, Pocket-a-2, Mayur Vihar, Phase 3, East Delhi, Kondli, DL | - | - |
| US | other | 0.496 | Bates's Audio Installation | Unit Ste 200, TX, Mckinney, 2740 Virginia Parkway | Spears Audio | 2740 Virginia Pkwy, # Ste 200, Mckinney, Texas | - | - |
| US | other | 0.862 | Chiropractic Modern Specialists L.L.C. | 180 Purtymun, Sedona, AZ | Chiropractic Modern Specialists Llc | 201 Purtymun, Sedona, Arizona | - | - |
| India | other | 0.961 | Nagya Agrotech Private Limited | 380, Sadar Bazar, Agra, Uttar Pradesh | Nagytis Agrotech Private Ltd | 3-80, SADAR BAZAR, AGRA, उत्तर प्रदेश | - | - |
| India | other | 0.999 | Kongunadu Engineering Private Limited | Wayanad, Ward No - 1X, Block No - 19, Valappil Complex, Gudlai, Road, Kerala, Kpa - Ix/205/14R, Kalpetta | Kongunadu-Engineering Limited | Kpa - Ix/205/15r, Ward No - 1X, Block No - 19, Valappil Complex, Gudlai, Road, Kalpetta, Wayanad, Keralam | - | - |
| India | common_name_crowd | 0.999 | Future Trading Private Limited | No. 2, Ramachandra Pura, Main Road, Jalahalli Port, Bangalore North, Bangalore, Karnataka | FUTURE TRADING PRIVATE LIMITED | #2/C, Bangalore, Bangalore North, ಕರ್ನಾಟಕ | Future Trading Private Limited | No 2/C, Santa Monica, Hayes Road, Shantala Nagar, Bangalore North, Bangalore, Karnataka |
| India | other | 0.896 | Ravin Technologies Pvt Ltd | Slv Tower, 878, 7Th Main Rd, Near Ragigudda Temple, Ksrtc Layout, 3Rd Phase, J. P. Nagar, Ground Floor, Bangalore South, Karnataka, Bangalore | Ravun Teecbnlogiles Pvt Ltd | GROUND FLOOR, SLV TOWER, 878, 7TH MAIN RD, NEAR RAGIGUDDA TEMPLE, KSRTC LAYOUT, 3RD PHASE, J. P. NAGAR, BANGALORE SOUTH, Karnataka | - | - |
| US | common_name_crowd | 0.896 | Continental Committee Inc. | 220 Post Oak Avenue, Rogers, TX | Continental  Committee Inc. | 229- POST OAK AVE, ROGERS CDP, TX | - | - |
| India | other | 0.926 | Vaayuva Management Private Limited | S. No. 370, Airstrip, Village Kathda, Mandvi, Kachchh, Gujarat | Vaayuvaer Maangement Private Limited | S. NO. 372, AIRSTRIP, VILLAGE KATHDA, MANDVI, Gujarat | - | - |
| US | common_name_crowd | 0.892 | Twisted Cafe! Inc. | 210 Washington Avenue, Elyria, OH | Twisted Cafe! Inc | OH, 215 WASHINGTON AVHNUE, ELYRIA | - | - | 

## FN examples

| country | mode | p_final | s1_name | s1_addr | r_name | r_addr |
|---|---|---|---|---|---|---|
| US | empty_address_record | 0.005 | Office of Transportation LLC | 511 Pearson Springs Road, Unit 820, Maryville, TN | Office of (Transportation) |  |
| US | empty_address_record | 0.231 | 9-Star PC | 6124 Algona Court, Fairfax County, VA | 9-Star PC |  |
| India | other | 0.694 | Pure Spinning Pvt Ltd | Door No 7/132-K, Porur Post Thaliyamkundu Wandoor, Malappuram, Malappuram, Malapuram, Kerala | Pure Smpnaing Pvt Ltd | #7/132-K. , PORUR POST THALIYAMKUNDU WANDOOR, MALAPPURAM, MALAPPURAM, കേരളം |
| US | empty_address_record | 0.015 | Blue Massage | 1371 Masseyville Road, Bethel Springs, TN | Blue Massage Ltd |  |
| US | empty_address_record | 0.054 | New Life Presbyterian Church | 1677 54th Avenue, TN, Nashville, Unit 233 | The New Life Presbyterian Church |  |
| US | empty_address_record | 0.625 | Verrex Mobile LLC | 1217 Logan, Unit 101, Belvidere, IL | Verrex Mobile L.L.C. |  |
| India | alias_or_web_name_form | 0.689 | Bipin Memorial Trust | 312 Kot Ni Pole Nr Talatihall Bhut Ni Ambali Raipur, Ahmedabad, Gujarat | mbtrust.com | 312 KOT NI POLE NR TALATIHALL BHUT NI AMBALI RAIPUR, AHMEDABAD, Gujarat |
| US | empty_address_record | 0.645 | Cybil Gust Associates Inc | 1157 Red Thimbleberry Drive, Elyria, OH | Cybil Gust Associates Inc |  |
| US | name_replaced_same_address | 0.304 | Halimeda Hyman Greenland Corp | 906 Shelley Road, Loch Raven, MD | Zetajaxquo | Maryland, Towson, 906 Shelley Rd |
| US | other | 0.497 | Miller & Canales | 4411 99th Avenue, Unit 2118, Phoenix, AZ | Miller + Cnhfle | 4411 99th Avenfe, # 2118, Phenix, Arizona |
| US | empty_address_record | 0.613 | Creative Pampa LLC | IA, 601 Aurora Avenue, Des Moines | Creative LLC Pampa |  |
| India | empty_address_record | 0.648 | Sard Technology Pvt Ltd | 512, Mint Street, Chennai-79. Chennai-79., Chennai-79., Chennai, Tamil Nadu | Sri Sard Téchnology Pvt Ltd |  |
| India | other | 0.520 | Pratibha Fit Limited | G.No.565/B, Near Birdev Temple Top, Karveer, Kolhapur, Maharashtra | Prtibta Fit Limited | G.eo.a-565/b, Near Birdev Temple Top, Karveer, Kolhapur, MH |
| US | empty_address_record | 0.101 | Alma Sawyer, MD | 604 Laura Lane, Two Harbors, MN | Alma Saewre, MD |  |
| US | empty_address_record | 0.379 | Scott Learning Center PLLC | 1614 Medford Avenue, Youngstown, OH | Scott Learning Center |  |
| US | empty_address_record | 0.379 | Adams, Carrillo and Coffman Corporation | 14124 Carrydale Avenue, Cleveland, OH | Adams, Carrillo and |  |
| US | empty_address_record | 0.172 | Patriot Learning Center | 1631 Peck Lane, Cheshire, CT | Patriot Léárning Center |  |
| India | empty_address_record | 0.154 | Shree Clinic Private Limited | 44A, Raja Basanta Roy Road, 2Nd Floor, Kolkata, Kolkata, Howrah, West Bengal | Shree Clinic Private Límited |  |
| US | empty_address_record | 0.262 | Grand Strategy LLC | 11513 Shelbyville Rd, KY, Middletown | Grand Strategy |  |
| India | empty_address_record | 0.120 | Chennai Marketing Pvt Ltd | Ground Floor, New No 16, Old No. 19 Jayalakshmi Puram 1St Street, Nungambakk, Am, Chennai, Tamil Nadu | Chennai Márketing Pvt Ltd |  |
| US | empty_address_record | 0.523 | Bowen River | 7071 Forsyth Boulevard, Fl 0, Saint Louis, MO | Bowen River [Ltd] |  |
| India | empty_address_record | 0.158 | Aashi India | C/O Mr. Kishan Kumar, S/O Hem Raj Chauhan, Faridabad, Haryana | Aashi Center |  |
| US | empty_address_record | 0.102 | Superior Mobility Holdings | 12113 Chapel Hill Road, El Paso County, TX | Superior Mobility |  |
| US | empty_address_record | 0.531 | Value Cargo Consultants | MD, 1000 Brightseat Road, Unit 111, Hyattsville | value cargo consultants |  |
| US | empty_address_record | 0.572 | Leboeuf Nexera LLC | 2110 Whispering Springs Road, Harrisonburg City, VA | Leb0euf Nexera LLC |  |
| India | native_script_name | 0.611 | Sky Logistics Private Limited | 02, Haryana, Bhondsi, Emaar The Palm Square, Sec-66 Golf Course Road, Gurgaon | Sky लॉजिस्टिक्स प्राइवेट लिमिटेड | 02, Palwal, Gurgaon, HR |
| India | empty_address_record | 0.210 | Paramount Traders Private Limited | H No. 5, Block - D, Landmark Gtk Road Sma Indl Area, New Delhi, Delhi, North West Delhi, Delhi | Paramount Traders Private |  |
| US | other | 0.567 | Beacon Network Global | 18018 97th Avenue Court, Puyallup, WA | Beacon Network 6lobal LLC | 97RD AVENUE COURT, PUYALLUP, WA |
| US | empty_address_record | 0.229 | The Dent Pub Center | 2001 Gemini Drive, Garland, TX | THE DENT PUB CENTER |  |
| US | other | 0.014 | Taylor, Fisher and Short Generating | 907 Fairland Street, Fl 1, Benton, IL | Taylor, Fisher and Short Generating Co | 908 FAIRLAND ST, BENTON, IL |
| US | empty_address_record | 0.153 | Supreme Heritage Associates | 815 Herman Road, Horsham Township, PA | Supreme Héritage |  |
| US | empty_address_record | 0.527 | Peridos Jersey Inc | 3906 Woodmont Drive, Houston, TX | Peridos Jérsey Inc |  |
| India | name_replaced_same_address | 0.724 | Simplex Techno Limited | 101, 1St Floor, Embassy Classic, 11 Vittal Mallya Road, Mahatma Gandhi Road, Bangalore North, Bangalore, Karnataka | ST | 101, 1St Floor, Embassy Classic, 11 Vittal Mallya Road, Mahatma Gandhi Road, Bangalore North, Bangalore, Karnataka |
| India | native_script_name | 0.533 | Ss Indian Consultants Private Limited | No 5/1, Kamarajar Nagar 10Th Street Korattur, Chennai, Tamil Nadu | எஸ்எஸ் இந்தியன் கன்சல்டன்ட்ஸ் பிரைவேட் லிமிடெட் | NO 005/1, CHENNAI, Tamil Nadu |
| US | empty_address_record | 0.149 | Katalin Baldridge, DO, P.C. | 5257 Barton Road, North Ridgeville, OH | Katalin Baldridge, DO,  P.C. |  |
| US | empty_address_record | 0.021 | Health Institute | 12402 Monroe Street, Avondale, AZ | Health Institute |  |
| US | empty_address_record | 0.329 | Reeves & Thompson L.L.C. | 479 Second Street, Albany, NY | Reeves + Thmpson L.L.C. |  |
| US | other | 0.491 | Cho Rio | 1038 Williamson Avenue, Burlington, NC | Cho Rio LLC | Williamson Ave, ELON, North Carolina |
| India | empty_address_record | 0.061 | Smart Foundation Limited | Tamil Nadu, Kanchipuram, 7Th Street Extension, Rajalakshmi Nagar, Tambaram, No.6 | SMART FOUNDATION LIMITED |  |
| US | empty_address_record | 0.124 | Zhu Continental Praetorian LLC | 152 Pakatakan Road, Middletown, NY | ZHU Continental Praetorian LLC |  | 
