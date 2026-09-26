# Error analysis

| kind | India | US |
|---|---|---|
| FN | 8479 | 11521 |
| FP | 4987 | 4117 |
| TP | 7850 | 12150 | 

## FN modes

| mode | India | US | total |
|---|---|---|---|
| empty_address_record | 2971 | 5077 | 8048 |
| similar_but_outranked | 2207 | 4030 | 6237 |
| common_name_crowd | 898 | 1251 | 2149 |
| name_replaced_same_address | 415 | 803 | 1218 |
| native_script_name | 957 | 0 | 957 |
| address_rewritten_name_kept | 503 | 8 | 511 |
| partial_name_and_address_noise | 252 | 70 | 322 |
| alias_or_web_name_form | 111 | 113 | 224 |
| no_key_survives_df_cap | 48 | 129 | 177 |
| name_replaced_address_changed | 117 | 40 | 157 | 

## FP modes

| mode | India | US | total |
|---|---|---|---|
| similar_but_outranked | 2294 | 2158 | 4452 |
| empty_address_record | 554 | 909 | 1463 |
| common_name_crowd | 621 | 604 | 1225 |
| native_script_name | 718 | 0 | 718 |
| name_replaced_same_address | 210 | 274 | 484 |
| address_rewritten_name_kept | 281 | 19 | 300 |
| partial_name_and_address_noise | 167 | 28 | 195 |
| alias_or_web_name_form | 72 | 70 | 142 |
| no_key_survives_df_cap | 36 | 49 | 85 |
| name_replaced_address_changed | 34 | 6 | 40 | 

## FP: true owner of the wrongly merged record

| belongs | India | US |
|---|---|---|
| another S1 | 1224 | 1203 |
| unmatched (no S1) | 3763 | 2914 | 

- H1: record contains a word of its true owner's name that our S1 lacks: **7.2%** of 2427
- owner S1 has exactly our S1's address (same-address different entity): **10.5%**
- owner name words == our name words (indistinguishable names): 55.9%
- H3: both have a house number and the first numbers differ: **44.9%** of FP (both-have-number 78.0%)
- FP record's exact address shared by >=2 S2/S3 records: 34.4%
- FP record's address == our S1 address (normalized): 3.9%

## FN: evidence from the S1's other true records

- H2: missed record has exactly the address of a sibling true record: **15.3%** of 20000
- missed record shares its first house number with a sibling: 35.2%
- missed record's address == our S1 address (normalized): 2.0%
- p1 of misses: median 0.334, share p1 > 0.3: 53.4%

## FP examples

| country | mode | p_final | s1_name | s1_addr | r_name | r_addr | owner_name | owner_addr |
|---|---|---|---|---|---|---|---|---|
| India | similar_but_outranked | 0.882 | Sarovar Cotton Limited | H/O Jagdish, Rajdev Gorakhpur, Jr No-2, Naibazar, Gorakhpur, Chauri Chaura, Gorakhpur, Uttar Pradesh | Sarovar Cotton Pvt Ltd | Rajdev Gorakhpur, Jr No-2, Naibazar, Gorakhpur, UP, No #31 H/o Jagdish, Ghazipur | - | - |
| India | empty_address_record | 0.901 | Shraddha Hospital | Aspera601 Lbs Marg, Raheja Gardens Thana, Thane, Maharashtra | Shraddha Hospital Private |  | Shraddha Hospital Private Limited | D-11/54, 1St Floor, Sector-7, Rohini, Delhi, North East, Delhi |
| India | common_name_crowd | 0.838 | Seven Systems Private Limited | Suite 5B, Sagas Amar Court, 59 G.N.Chetty Road, T.Nagar., Chennai, Tamil Nadu | Seven Systems Public Limited | SUITE 18B., SAGAS AMAR COURT, 59 G.N.CHETTY ROAD, T.NAGAR., CHENNAI, Tamil Nadu | - | - |
| US | similar_but_outranked | 0.837 | Stokes Holdings | 4 Coriander Drive, Moreau, NY | Stokes Holdings Co | 15 CORIANDER DRIVE, FORT EDWARD, NY | - | - |
| India | empty_address_record | 0.974 | Vsg (India) Lab | Cts E 108 109 A Wing Neelkamal, K M Colony 13Th Road Khar West, Mumbai, Maharashtra | Vsg (India)  Lab |  | Vsg (India) Lab | Srivari Srimath 3Rd Floor Door No.1045 Avinashi Road, Coimbatore, Tamil Nadu |
| India | similar_but_outranked | 0.858 | Agrani Clinic | C/O Ramdas Pansare, Pl No, 12, A.S.E. Ashwin Sector, Nashik, Maharashtra | *** AGRANI CLINIC INFRATECH COMPANY | NASHIK, PL NO, 12, A.S.E. ASHWIN SECTOR, NO 29 C/O RAMDAS PANSARE, Maharashtra | - | - |
| India | similar_but_outranked | 0.873 | Healthcare Equal Projectsprivate Private Limited | Marketopper House, 848, Udyog Vihar Phase - V, Gurgaon, Haryana | HEALTHCARE EQUAL PROJECTSPRIVATE LIMITED | Gurgaon, No 88 Marketopper House, HR | - | - |
| US | empty_address_record | 0.897 | Summit Strategic Physical Therapy | 523 Winthrop Drive, Southold, NY | Summit Strategic Physíca1 Therapy |  | Summit Strategic Physical Therapy | Osage City, KS, 322 12th Street |
| India | native_script_name | 0.927 | Ss Investments LLP | C/O Dhondiram Patil M.No -2477 A/P - Bhadole, Kolhapur, Maharashtra | एसएस सिस्टम्स एलएलपी | C/o Dhondiram Patil M.no -2477 A/p - Bhadole, Kolhapur, MH | - | - |
| India | common_name_crowd | 0.768 | One Global Private Limited | No:1, 2Nd Cr, Sarvabhowma Nagar B.G Road, Bangalore, Karnataka | ONE GLOBAL PVT  LTD. | #4/1, 2ND CROSS, B.G ROAD, BANGALORE, Karnataka | One Global Pvt Ltd | 2Nd Cross, B.G Road, Bangalore, Karnataka, 4/1, Bangalore |
| India | similar_but_outranked | 0.810 | Thrissur Guest Private Limited | Door No 10/176Pappachan Building P O Pavaratty, Thrissur, Kerala | Thrissur Engineers Private | DOOR NO 10/183PAPPACHAN BUILDING P O PAVARATTY, THRISSUR, Kerala | - | - |
| US | similar_but_outranked | 0.750 | Elle Oman, M.D., P.C. | 16 Pond Street, Unit 1, Holbrook, MA | ELLE OMAN, M.D., P.C. INC | 37 POND SAINT, HOLBROOK, MA | - | - |
| India | empty_address_record | 0.786 | Rose (India) Services Private Limited | L906, Aparna Serene Park, Masjidbanda Road Kondapur, Serilingampally, Hyderabad, Telangana | Rose (India) Srbvides Private Limited |  | Rose (India) Services Private Limited | Vill Gopalpur Uttarpara, Durgapur Dist- Burdwan, Durgapur, Bardhaman, West Bengal |
| India | similar_but_outranked | 0.905 | Rainbow Trading Private Limited | Quila Chowk, Faridkot, Punjab | Rainbow Trading Limited | ਪੰਜਾਬ, Faridkot, No 50 Quila Chowk, Faridkot | - | - |
| US | name_replaced_same_address | 0.524 | Ranion Frontier Fusion Inc | Bono, 3527 Craighead 323 Road, AR | ZP | Bono, AR, 3527 Craighead 323 Road | Zephara Purpose | 3527 Craighead 323 Road, Bono, AR |
| US | similar_but_outranked | 0.827 | Innovative Genomic Industries Partners | 5398 Old Post Road, Ogden, UT | Innovative  Genomic Industries Partners Inc | 2403 OLD POST RD, OGDDEN, UT | - | - |
| India | similar_but_outranked | 0.987 | Nigam Consultants Limited | 84/1A, Topsia Road South Trinity Plaza, 1St Floor, Kolkata, Kolkata, Howrah, West Bengal | Nigam Consultants Private Limited | 84/10A, Topsia Road South Trinity Plaza, 1St Floor, Kolkata, Calcutta, Howrah, WB | - | - |
| India | common_name_crowd | 0.886 | Sunrise Producer | Classic Weigh Bridge Thottakkattukara, Aluva, Ernakulam, Kerala | SUNRISE PRODUCER LIMITED | No 48- Classic Weigh Bridge Thottakkattukara, Ernakulam, KL | - | - |
| US | similar_but_outranked | 0.845 | TKS Summit Four | 3952 Crosswinds Drive, Rocky Mount, NC | TKN Summit Four | 3952 CROSSWINDS DRIVE, NC, ROCKY MOUNT | - | - |
| India | similar_but_outranked | 0.980 | Thane Trade Ltd | B306, Radhakrishana Behind, Hotel, 150 Feet Road, Maharashtra, Thane | Thane Trade Pvt Ltd | MH, #975 B308, Radhakrishana Behind, Hotel, 150 Feet Road, Thane | - | - |
| India | similar_but_outranked | 0.768 | Venba Traders Pvt. Ltd. | H No : 6-3-179/16, Jai Nagar Colony New Bhoiguda, Secunderabad, Hyderabad, Telangana | Venba Traders Exports Pvt | No 275 H No : 6-3-179/25, Hyderabad, TG | - | - |
| US | similar_but_outranked | 0.946 | Verric Jena Inc | 1209 Olde Oaks Drive, Zachary City, LA | Jena Verrik Inc | 1209 Olde Oaks Dr, Zachary City, Louisiana | - | - |
| India | native_script_name | 0.963 | Galaxy Impex Private Limited | Fl No 8 Pl No 2 Sr No 35 Shanti Elite Near Wisdom High School Nashik, Nashik, Maharashtra | गैलेक्सी इम्पेक्स टेक्नोलॉजीज प्राइवेट लिमिटेड | No 33 Fl No 8 Pl No 2 Sr No 35 Shanti Elite Near Wisdom High School Nashik, Nashik, MH | - | - |
| US | common_name_crowd | 0.971 | Geia | 414 Summit Drive, Blades, DE | Geia Corp | 823 SUMMIT DRIVE, <NULL>, BLADES, DE | - | - |
| US | common_name_crowd | 0.973 | Heritage Foundation Inc | Temple Hills, MD, Unit 301, 3452 Brinkley Road | heritage foundation co | 345 Brinkley Rd, Unit 301, Temple Hills, Maryland | - | - |
| India | similar_but_outranked | 0.909 | Moonshine Group of Companies Limited | 33/12A Nagalingam Street, Arisipalayam, Salem, Tamil Nadu | Moonshine Group of LLP Service | #33/13 NAGALINGAM STREET, ARISIPALAYAM, SALEM, தமிழ்நாடு | - | - |
| US | common_name_crowd | 1.000 | Empire Animal Hospital | 4512 Kubala Store Road, Yorktown, TX | Empire Animal Hospital Co | 4512 KUBALA STORE RD, YORKTOWN, TX | - | - |
| US | common_name_crowd | 0.772 | 6-Corners | Unit 204, Washington, DC, 226 R Street | 6 Corners Co | 239. R St, # 204, Washington, District of Columbia | - | - |
| India | similar_but_outranked | 0.979 | Grand (India) Infrabuild Private Limited | Plot No. 9A, D. No. 7/42, Asthinapuram Road, Drb Nagar, Nanmangalam, Old Pallavaram, Kancheepuram, Tamil Nadu | Limited Grand (India) Infrabuild | 18A, D. No. 7/42, Asthinapuram Road, Drb Nagar, Nanmangalam, Old Pallavaram, Kancheepuram, தமிழ்நாடு | - | - |
| India | native_script_name | 0.824 | Leo Management Private Limited | Hyderabad, Telangana, Flat In Adi Flat No 303, 2Nd Floor Plot No 6-3-1111/7, Begumpet, Hyderabad | మై మేనేజ్‌మెంట్ ప్రైవేట్ లిమిటెడ్ | FLAT NO 303, HYDERABAD, Telangana | My Management Private Limited | Flat No 303, Everest, Halcyon, Road No.78, Padmalaya Studios, Jubilee Hills, Shaikpet, Hyderabad, Telangana |
| India | partial_name_and_address_noise | 0.999 | Indrin India Pvt Ltd | Office - 402, Sitabag Colony, Sn - 124/3, Shivranjani House/ Dattawadi, Pune, Maharashtra | Indrina Inci Pvt Ltd #22221 | Office - 402, महाराष्ट्र | - | - |
| US | similar_but_outranked | 0.982 | UL Interstate Taiwan Corporation | 734 Ivy Creek, Unit APT 3, Ivel, KY | -- Powell Interstate Taiwan (Corporation) | 734 IVY CREEK, IVEL, KY | - | - |
| India | native_script_name | 0.929 | Supreme Developers Private Limited | Plot-46, Khasra-64, Begariya, Dubagga, Lucknow, Uttar Pradesh | सुप्रीम डेवलपर्स प्राइवेट लिमिटेड | D-##169/46, Noida, UP | Supreme Developers Private Limited | Sector-50, Uttar Pradesh, Noida, Gautam Buddha Nagar, D-169/46 |
| India | empty_address_record | 0.876 | Sagar Public School | # E-604, Shriram Samruddhi Apts, Varthur Road, Thubrahalli, Bangalore, Karnataka | Sagar Public Schóol |  | Sagar Public School | 187A, Ground Floor, Garud Apartment Pocket 4, Mayur Vihar, Phase-1, Noida, New Delhi, Delhi |
| India | similar_but_outranked | 0.812 | Travels Plantations Private Limited | C/O Kaloo Singh, Water Works Colony Sikandra Road, Bandikui, Dist Dausa, Bandikui, Jaipur, Rajasthan | Travels Plantations Limited | 20 C/O KALOO SINGH, WATER WORKS COLONY SIKANDRA ROAD, BANDIKUI, DIST DAUSA, BANDIKUI, राजस्थान | - | - |
| India | empty_address_record | 0.836 | Pioneer Constructions Private Limited | B-12Greater Kailash Enclave-I, New Delhi, South Delhi, Delhi | Pioneer Constructions Private [Limited] |  | Pioneer Constructions Private Limited | 86 L Road Bhopalpura, Udaipur, Rajasthan |
| India | native_script_name | 0.925 | Gold Systems Private Limited | 47-B Lajpat Nagar Ii, Delhi, South Delhi, Delhi | गोल्ड आईटी लिमिटेड | 47, South West Delhi, Delhi, DL | Gold It Limited | 47, Ground Floor, Shakti Vihar, Pitam Pura, Delhi, South West Delhi, Delhi |
| India | similar_but_outranked | 0.784 | Sudarshan College | R1-E Bldg, Flat 705, Life Republic Mulshi Pune, Mulashi, Pune, Maharashtra | Sudarshan Ventures | R5-E BLDG, FLAT 705, LIFE REPUBLIC MULSHI PUNE, MULASHI, महाराष्ट्र | - | - |
| US | empty_address_record | 0.888 | Colburn and Hammack L.L.C. | 132 Front Street, Drummond, MT | COLBURN AND [HAMMACK] |  | Colburn and Hammack L.L.C. | 157 NW Backwoods Road, Moyock, NC |
| India | empty_address_record | 0.961 | Alliance Hospital | H No. 24, Gali No. 12, Ajit Nagar, Jalandhar - I, Jalandhar, Punjab | Alliance Hospital |  | Alliance Hospital | Maharashtra, Mumbai, Oberoi Garden Estate, Chandivali, B-1063, Andheri (E). | 

## FN examples

| country | mode | p_final | s1_name | s1_addr | r_name | r_addr |
|---|---|---|---|---|---|---|
| US | empty_address_record | 0.051 | Avivah Dunn Vertical | ND, 609 3rd Street, Unit Apartment 4, Fargo | Avivah Vertical Dunn |  |
| India | native_script_name | 0.640 | Fortune Impex Private Limited | C-16, Moti Marg, Bapu Nagar Jaipur, Jaipur, Rajasthan | फॉर्च्यून इम्पेक्स प्राइवेट लिमिटेड | JAIPUR, Rajasthan, #803 C-16 |
| US | empty_address_record | 0.021 | Piedmont Committee | 27 Avalon Gardens Drive, Clarkstown, NY | Piedm0nt [Committee] |  |
| US | similar_but_outranked | 0.456 | Safe Bay Artificial, LLC | 4201 Augusta Way, Gresham, OR | Safe Artificia1, LLC Services | 7201 Augusta Way, <NULL>, Gresham, Oregon |
| US | empty_address_record | 0.020 | Blue Association | N7810 Parkway Road, Town Of Stephenson, WI | Blue Association Inc. |  |
| India | empty_address_record | 0.673 | Astrum Hospitality LLP | 135, Jodhpur Park Jodhpur Park, Kolkata, Kolkata, Howrah, West Bengal | Astrum Hospitality L.L.P. |  |
| India | name_replaced_same_address | 0.434 | OVR Constructions | No. 145, Gothavari Street, Palaniappa Nagar, Valasaravakkam, Chennai, Tamil Nadu | Iriiri | TN, Chennai, Gothavari Street, Palaniappa Nagar, Valasaravakkam, Chennai, No. 1-45 |
| US | name_replaced_same_address | 0.397 | 1 800 I Care 4 U | 1694 Park Preserve Way, Unit 17, Freeport, IL | #1800 | 7694 PARK PRESERVE WAY, FREEPORT, IL |
| US | empty_address_record | 0.022 | Frontier Society LLC | 717 Union Street, Warren, OH | The Frontier Society LLC |  |
| US | common_name_crowd | 0.177 | Cedar LLC | 3817 Rice Road, Tahlequah, OK | Cedar-LLC | 3681 RICE ROAD, TAHLEQUAH, OK |
| US | empty_address_record | 0.129 | Crandall Enterprises Inc | 307 9th Street, Watkins Glen, NY | Crandall (Enterprises) |  |
| India | empty_address_record | 0.071 | Silver Infrastructure Private Limited | B-30/267 Sector 51 Noida, Noida, Gautam Buddha Nagar, Uttar Pradesh | SILVER INFRASTRUCTURE PRIVATE [LTD] |  |
| India | empty_address_record | 0.546 | Dhara Sangh | North 24 Parganas, Akanksha Housing, West Bengal, New Town, Cl-2/306, Pl-Db2, St-300, Kolkata | Dhara Sangh Ltd |  |
| US | similar_but_outranked | 0.544 | Value Cloud Group | 1645 91st Street, Clive, IA | Value Group-Services | 1645 91ST STREET, CLIVE, IA |
| India | partial_name_and_address_noise | 0.428 | Faridabad Spirits | House No 387, Ground Floor, Sector 46, Faridabad, Haryana | Faridabad 5ervice | Faridabad, हरियाणा, Faridabad, Plot 11 House No 387 |
| India | address_rewritten_name_kept | 0.607 | Bangalore Marketing Private Limited | B1101, Snn Raj Etternia Off Haralur Road Parappana, Agrahara, Silvercounty Road, Bangalore, Karnataka | Bangalore Márketing Private Limited | Blsck F-981- B1101, Bangalore, KA |
| US | similar_but_outranked | 0.664 | Applied Aerospace Worldwide LLC | 1002 Dogwood Drive, Star City, AR | Applied Aerospace Worldwide Worldwide LLC | 100 Dogwood Drive, Star City, Arkansas |
| India | name_replaced_same_address | 0.240 | Rejoice Law Chambers | 106, Ijmima Complex, Behind Infinity Mall Link Road, Malad West, Mumbai, Mumbai City, Maharashtra | Mr Yumalyra | Plto G-806 106, Ijmima Complex, Behind Infinity Mall Link Road, Malad West, Mumbai City, NULL, MH |
| US | similar_but_outranked | 0.757 | Surgical Trusted Physicians Inc | 11897 Marlowe Drive, Dakota Ridge, CO | Surgical Trusted Pahysciis Inc | 11897d Marlowe Drive, Dakota Ridge, Colorado |
| India | empty_address_record | 0.326 | Track Enterprises LLP | 65, Subhas Nagar B Jodhpur Road, Pali, Rajasthan, Pali Marwar | Mr Track Enterprises  LLP |  |
| India | address_rewritten_name_kept | 0.347 | Usman Medical Centre Private Limited | 118, Uuf, Prakashdeep Building 7, Tolstoy Marg, New Delhi, Delhi | Private Usman Medical Cte Limited | 1-18, NEW DELHI, Delhi |
| India | similar_but_outranked | 0.670 | Leather & Brothers Private Limited | Madhuranjan 1277 Jangali Maharaj Road, Pune, Maharashtra | Leather & 8rothers Brothers Private | Madhuranjan 12-77 Janglai Maharaj Road, Pune, MH |
| US | similar_but_outranked | 0.581 | Diamond Supply Brands Inc | 200 The Woods, Bedford, IN | Inc Diamond Supply Brands | 186 THE WOODS, BEDFORD, IN |
| US | name_replaced_same_address | 0.575 | Behavioral Health Group of Lake Ozark PC | 114 Tara Road, Unit 1A, MO, Lake Ozark | HEALTHLAKEBEHAVIORALCOM | 114 TARA RD, LAKE OZARK, MO |
| US | empty_address_record | 0.265 | Creative Printing Labs Inc. | 49825 Gallatin Road, Gallatin Gateway, MT | Creative Printing Labs |  |
| India | empty_address_record | 0.018 | Urban Investment Private Limited | Maharashtra, Lohkare Gondegaon, C/O Balkrishna Natthuji, Darwha, Yavatmal | Urban Investment-Private Limited |  |
| India | similar_but_outranked | 0.161 | Solutions Udaan Commodities LLP | Kc-09, Shilpi Green Colony, Chirhula Road, Badraon, Huzur, Rewa, Madhya Pradesh | Solutions Udaan LLP Center | NO. 313 KC-09, SHILPI GREEN COLONY, CHIRHULA ROAD, BADRAON, HUZUR, Madhya Pradesh |
| India | similar_but_outranked | 0.027 | Mck Welfare Society | West Bengal, Kolkata, Ground Floor. R No 13, 2 Clive Ghat Street, Howrah | SOCIETY MCK WELFARE | H.NO 5-2 CLIVE GHAT STREET, null, CALCUTTA, HOWRAH, West Bengal |
| US | empty_address_record | 0.523 | North Shreya LLC | 16 Hillside Drive, Peru, NY | North Shreya [LLC] |  |
| India | partial_name_and_address_noise | 0.106 | OA Global Private Limited | Khasra No 186/1Rajeev Col, Ony Sector-56 Ballabgarh, Ballabgarh, Faridabad, Haryana | OA Private Limited Partners | ###1879 Col, Ony Sector-56 Ballabgarh, Ballabgarh, Ballabhgarh, हरियाणा |
| US | name_replaced_same_address | 0.481 | Debt LLC | 241 Liberty Street, Dendron Town, VA | *** De LLC LLC | 241A Liberty St, Dendron, Virginia |
| US | empty_address_record | 0.570 | Johnson Aldabra Corp | 2921 Main Street, Unit Unit A, Parish, NY | Johnson Aldabra |  |
| US | empty_address_record | 0.601 | Pinnacle Etf P.C. | 2151 Momany Street, Oregon, OH | Pinnacle Etf  P.C. |  |
| India | native_script_name | 0.535 | Guru Properties Pvt Ltd | 7/70, Om Sai Pragati Co-Op. Hsg. Soc. Ltd., Mhb Colony, Mahavir Nagar, Kandivali (We, St), Mumbai, Mumbai City, Maharashtra | गुरु Properties प्रा. लि. | 70, Mumbai City, MH |
| India | empty_address_record | 0.342 | Afzal Tourist Corporation | D 1901, Megapolis, Hinjewadi, P-3, Haveli, Pune, Maharashtra, Sangria | AFZAL  TOURIST CORPORATION |  |
| India | common_name_crowd | 0.483 | Blue Consultants | A-1905, 19Th Floor, Tower 15, Purvanchal Royal City, Sector Chi-5, Greater Noida, Noida, Gautam Buddha Nagar, Uttar Pradesh | BLUE  CENTER | DOOR NO #111 A-1905, 19TH FLOOR, TOWER 15, PURVANCHAL ROYAL CITY, SECTOR CHI-5, GREATER NOIDA, NOIDA, उत्तर प्रदेश |
| US | empty_address_record | 0.452 | Orthopedic Direct Medicine Inc | 140 Chapel Road, South Windsor, CT | Orthopedic-Direct Medicine  Inc |  |
| India | similar_but_outranked | 0.678 | OS Solutions Limited | Ahana Business Centre, Plot No-10, D No-4/983 Gandhi St, (Omr Road), Nehru Nagar, Kott, Ivakkam, Chennai, Tamil Nadu | OS-Solutions | CHENNAI CITY REGION, CHENNAI, #520 AHANA BUSINESS CENTRE, Tamil Nadu |
| India | similar_but_outranked | 0.629 | Ramya India Pvt. Ltd. | A-217 Block A, Dairy Farm, Gharoli, Delhi, East Delhi, Delhi | Ramya Ramya Pvt. Ltd. Services | PLOT 43 A-217 BLOCK A, DAIRY FARM, GHAROLI, DELHI, Delhi |
| US | similar_but_outranked | 0.599 | Athey, Johnson & Binder Aspen PC | Paris, TN, 601 Russell Street | Athey, Johnson & Binder Aspen PC | PARIS, TN, 235 RUSSELL ST | 
