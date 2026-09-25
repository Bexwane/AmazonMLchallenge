# Error Buckets (full-density blocker)

Blocker: `E003-style k_comb=45 k_name=10 df_cap=2500`. True pairs: 7,638,365. Missed (absent from the candidate set): 1,061,608 (13.90%). Found pairs compared against a random sample of 300,000 found pairs.

## Present vs absent (mandatory split)

| bucket | true_pairs | share | share_India | share_US |
|---|---|---|---|---|
| absent from candidate set | 1061608 | 0.139 | 0.207 | 0.094 |
| present, rank 1-10 | 6099840 | 0.799 | 0.721 | 0.850 |
| present, rank 11-20 | 234082 | 0.031 | 0.031 | 0.030 |
| present, rank 21-30 | 104795 | 0.014 | 0.016 | 0.012 |
| present, rank > 30 | 138040 | 0.018 | 0.025 | 0.013 |
| present but outside top-10 | 476917 | 0.062 | 0.072 | 0.056 |
| present but outside top-20 | 242835 | 0.032 | 0.041 | 0.026 |
| present but outside top-30 | 138040 | 0.018 | 0.025 | 0.013 |

A reranker can only recover the *present* rows; *absent* rows need new retrieval.

## Absent pairs: mechanism

* no key survives the df cap (name_cos + addr_cos = 0): 18.14% of absent
* scored but outranked: 81.86% of absent
* full-pool rank of scored-but-absent pairs (sample of 11,303): {'0.1': 2.0, '0.25': 49.0, '0.5': 84.0, '0.75': 257.0, '0.9': 747.0}

## Top-10 measured failure modes (ordered rules, first match wins, partition of absent pairs)

| failure_mode | missed_pairs | share_of_misses | country | source |
|---|---|---|---|---|
| native_script_name | 369850 | 0.348 | {'India': 1.0} | {'S2': 0.628, 'S3': 0.372} |
| common_name_crowd | 294212 | 0.277 | {'US': 0.619, 'India': 0.381} | {'S3': 0.563, 'S2': 0.437} |
| similar_but_outranked | 106338 | 0.100 | {'US': 0.721, 'India': 0.279} | {'S3': 0.534, 'S2': 0.466} |
| empty_address_record | 105323 | 0.099 | {'US': 0.642, 'India': 0.358} | {'S3': 0.522, 'S2': 0.478} |
| alias_or_web_name_form | 66004 | 0.062 | {'US': 0.609, 'India': 0.391} | {'S3': 0.575, 'S2': 0.425} |
| name_replaced_same_address | 53508 | 0.050 | {'US': 0.687, 'India': 0.313} | {'S3': 0.579, 'S2': 0.421} |
| no_key_survives_df_cap | 27418 | 0.026 | {'US': 0.794, 'India': 0.206} | {'S3': 0.54, 'S2': 0.46} |
| address_rewritten_name_kept | 14448 | 0.014 | {'India': 0.995, 'US': 0.005} | {'S3': 0.955, 'S2': 0.045} |
| partial_name_and_address_noise | 13633 | 0.013 | {'India': 0.861, 'US': 0.139} | {'S3': 0.834, 'S2': 0.166} |
| name_replaced_address_changed | 10874 | 0.010 | {'India': 0.799, 'US': 0.201} | {'S3': 0.593, 'S2': 0.407} |

## Flags: share among misses vs among found pairs

| flag | share_of_misses | share_of_hits | lift |
|---|---|---|---|
| common_name (freq>=20) | 0.722 | 0.190 | 3.808 |
| name_tok_jac<0.5 | 0.636 | 0.165 | 3.850 |
| source_S2 | 0.501 | 0.480 | 1.043 |
| addr_tok_jac>=0.75 | 0.400 | 0.658 | 0.608 |
| r_native | 0.349 | 0.028 | 12.602 |
| name_tok_jac=0 | 0.349 | 0.085 | 4.101 |
| last_component_eq (state/region) | 0.264 | 0.349 | 0.758 |
| r_has_no_number | 0.202 | 0.134 | 1.513 |
| name_skel_jac=0 | 0.193 | 0.080 | 2.406 |
| no_surviving_key (comb=0) | 0.181 | 0.000 | inf |
| addr_tok_jac<0.25 | 0.158 | 0.048 | 3.265 |
| r_addr_tokens<=3 | 0.118 | 0.040 | 2.930 |
| r_addr_empty | 0.099 | 0.035 | 2.842 |
| rare_name (freq<=1) | 0.071 | 0.116 | 0.613 |
| num_jac=0 (both have numbers) | 0.070 | 0.044 | 1.592 |
| name_compact_eq | 0.070 | 0.680 | 0.102 |
| r_alias_or_web | 0.063 | 0.084 | 0.758 |
| s1_name_tokens>=4 | 0.053 | 0.117 | 0.454 |
| s1_name_tokens=1 | 0.031 | 0.028 | 1.112 |
| s1_name_chars<=6 | 0.022 | 0.021 | 1.075 |
| postcode_both | 0.019 | 0.057 | 0.337 |
| postcode_eq | 0.016 | 0.053 | 0.298 |

## By country / source / script

| country | misses | share_of_misses | share_of_hits |
|---|---|---|---|
| India | 632394 | 0.596 | 0.371 |
| US | 429214 | 0.404 | 0.629 |

| source | misses | share_of_misses | share_of_hits |
|---|---|---|---|
| S2 | 531408 | 0.501 | 0.480 |
| S3 | 530200 | 0.499 | 0.520 |

| r_script | misses | share_of_misses | share_of_hits |
|---|---|---|---|
| Bengali | 24112 | 0.023 | 0.002 |
| Devanagari | 206026 | 0.194 | 0.016 |
| Gujarati | 23838 | 0.022 | 0.002 |
| Gurmukhi | 5455 | 0.005 | 0.000 |
| Kannada | 30339 | 0.029 | 0.002 |
| Latin | 690902 | 0.651 | 0.972 |
| Malayalam | 14958 | 0.014 | 0.001 |
| Oriya | 5731 | 0.005 | 0.000 |
| Tamil | 28354 | 0.027 | 0.002 |
| Telugu | 31799 | 0.030 | 0.002 |
| other | 94 | 0.000 | 0.000 |

## Examples (5 per mode)

### native_script_name

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| India | S3 | Ss Food Private Limited | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | एसएस फूड प्राइवेट लिमिटेड | Af-684, Ghaziabad, UP | 0.000 | 0.622 |
| India | S2 | Ss Food Private Limited | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | एसएस फूड प्राइवेट लिमिटेड | AF-0684, Uttar Pradesh, GHAZIABAD, 9487203 | 0.000 | 0.780 |
| India | S2 | Hotel Enterprises Limited | Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi | होटल एंटरप्राइजेज लिमिटेड | WZ-187C SHOP NO.13, DELHI, WEST DELHI, Delhi | 0.000 | 0.731 |
| India | S2 | Balaji Investment Private Limited | Plot No. D-88 & D-90, Hyd, Telangana, Hyderabad, Jeedimetla | బాలాజీ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్ | H.NO 00516 PLOT NO. D-88 & D-90, JEEDIMETLA, HYD, HYDERABAD, Telangana | 0.000 | 0.587 |
| India | S2 | Swastik Om Solutions LLP | 201/D, Aditya, Svp Nagar Andheri (W), Mumbai, Mumbai City, Maharashtra | स्वस्तिक ॐ सॉल्यूशंस एलएलपी | 201/D, BHIWANDI, MUMBAI CITY, Maharashtra | 0.000 | 0.000 |

### common_name_crowd

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| India | S3 | Hotel Enterprises Limited | Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi | Hotel Limited Services | Block B-517 Wz-187c Shop No.13, Divreportingcircle, West Delhi, DL | 0.000 | 0.482 |
| US | S3 | Obsidian, LLC | 3907 Hamilton Road, Deer Park, WA | Obsidian, Llc | Deer Park, Washington, Hamilton Rd | 1.000 | 0.000 |
| India | S3 | Balaji Investment Private Limited | Plot No. D-88 & D-90, Hyd, Telangana, Hyderabad, Jeedimetla | Balaji Insvstemnt Private Limited | Plot No. D-88 & D-90, Hyd, Shapurnagar, TG | 0.000 | 0.435 |
| US | S3 | Cozy Grill | 617 Fourth Street, Watseka, IL | Cozy Gle | 617 4th Street, Watseka, Illinois | 0.000 | 1.000 |
| US | S2 | Physical Therapy Clinic | 5208 Beacon Falls Drive, Fl 1, Columbia, MO | PHYSlCAL THERAPY CLINIC | BEACON FALLS DR, COLUMBIA, MO | 0.000 | 0.000 |

### similar_but_outranked

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| US | S3 | Tiena L. Hamilton, DDS | 31415 Orchard Hill Lane, Spring, TX | dds l. hamilton, tiena | Orchard Hill Lane, Spring, Texas | 0.223 | 0.000 |
| US | S2 | Beacon Municipals LLC | 15602 60, Borden, IN | Beacon Mumnicilpas LLC | 1560 60, BORDEN, IN | 0.000 | 0.222 |
| US | S2 | Modern Soulpower LLC | 1002 Kensington Circle, Fredericksburg City, VA | Modern  LLC Center | KENSINGTON CIRCLE, FREDERICKSBURG CITY, VA | 0.000 | 0.578 |
| US | S3 | Green Staffing Group Group | Springfield, IL, 41 Groton Drive | Green Group Group Services | 41- Groton Dr, Springfield, Illinois | 0.000 | 0.421 |
| US | S3 | Vierra and Yahn Empire | 1543 173rd Street, Hammond, IN | Vierra and Empire Yahn | 543 173rd Street, Hammond, Indiana | 0.712 | 0.000 |

### empty_address_record

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| US | S2 | Orellana Investments LLC | 728 A Quail Avenue, Fl Ground Floor, Geneva, IA | Orellana Investments Investments Llc |  | 0.462 | 0.000 |
| US | S2 | Obsidian, LLC | 3907 Hamilton Road, Deer Park, WA | obsidian, llc |  | 1.000 | 0.000 |
| US | S2 | Summit Health LLC | 95 Forest Edge Drive, Eads, TN | Summit Health Enterprises |  | 0.000 | 0.000 |
| India | S3 | Red Consultants Pvt. Ltd. | C/O Tapas Kumar Betal, Bhogpur, Purba Medinipur, Panskura, East Midnapore, West Bengal | Red Pvt. Ltd. Center |  | 0.000 | 0.000 |
| US | S2 | Physical Therapy Clinic | 5208 Beacon Falls Drive, Fl 1, Columbia, MO | Physical [Clinic] Therapy |  | 0.000 | 0.000 |

### alias_or_web_name_form

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| US | S2 | Cozy Grill | 617 Fourth Street, Watseka, IL | gcozy.com | 617 FOURTH ST, WATSEKA, IL | 0.000 | 1.000 |
| US | S3 | Tri-State Guild | 2915 Dares Beach Road, Prince Frederick, MD | tristateguild.com | 915 Dares Beach Rd, PMB 6098, Prince Frederick CDP, Maryland | 0.489 | 0.410 |
| US | S2 | Historical Committee Inc. | 3823 Hidden Cove Court, Rockwall, TX | HÍSTORICALCOMMITTEE.COM | HIDDEN COVE CT, PMB 4364, ROCKWWALL, TX | 0.497 | 0.000 |
| US | S3 | Pediatric Dental Medicine PC | 4735 Heber Springs Road, Ida, AR | dpmedicine.com | 4735 Heber Springs Road, Ida, Arkansas | 0.000 | 1.000 |
| US | S3 | Physical Therapy Group Inc | 7311 Madison Commons Lane, Houston, TX | Physicaltherapygroup.Com | Texas, 7311 Madison Commons Lane, Houston | 0.486 | 0.440 |

### name_replaced_same_address

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| US | S3 | Maure Williams Colombier Inc | 85 Wayne Avenue, Ticonderoga, NY | Dréxkor | 85 Wanye Avenue, Ticonderoga Townshiip, New York | 0.000 | 0.384 |
| US | S3 | Vadyne Inc | Greensboro, 1604 Washington Street, NC | Vagdyae Inc | 01604 Washington Street, PMB 841, Greensboro, North Carolina | 0.000 | 0.221 |
| India | S3 | Wave & Brothers Ltd | Plot No. 65, Mahada Colony, Sai Nagar, Nagpur, Maharashtra | Belobrixlum | Door No 65, Mahada Colony, Sai Nagar, Nagpur, MH | 0.000 | 0.518 |
| US | S3 | Green Staffing Group Group | Springfield, IL, 41 Groton Drive | greenstaffinggroupcom | 41- Groton Drive, Springfield, Illinois | 0.150 | 0.421 |
| US | S2 | VWQ Brookfield LLC | Unit Apartment 1, 52 Losson Road, NY, Cheektowaga | VB | BUFFALO CITY, NY, 52 LOSSON ROAD | 0.000 | 0.414 |

### no_key_survives_df_cap

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| US | S3 | Keystone Tri-State Ethereum LLC | 1304 Woodland Street, Springfield, MO | Keystone Tri-State | 1297- Woodland Street, Springfield, Missouri | 0.000 | 0.000 |
| US | S3 | Internal Medicine Silver Care | 129 Geneva Road, New Bern, NC | Internal  Care Silver Medicine | North Carolina, New Bern, 129 Geneva Rd | 0.000 | 0.000 |
| US | S2 | Johnson Holdings Group LLC | Saint Paul, MN, 5300 East Street | JOHNSON HOLDINGS | 300 EAST ST, PMB 3724, SAINT PAUL, MN | 0.000 | 0.000 |
| US | S2 | Johnson Holdings Group LLC | Saint Paul, MN, 5300 East Street | Jhsnno Holdings Group LLC | SAINT PAUL, EAST ST, MN | 0.000 | 0.000 |
| US | S2 | Modern Precision Laboratories | 10824 Edgewood Road, Harrison, OH | Services Laboratories Modern | 1082 EDGEWOOD RD, HARRISON, OH | 0.000 | 0.000 |

### address_rewritten_name_kept

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| India | S3 | Swastik Om Solutions LLP | 201/D, Aditya, Svp Nagar Andheri (W), Mumbai, Mumbai City, Maharashtra | Swastik Om LLP Service | 01/D, Mumbai City, महाराष्ट्र | 0.000 | 0.000 |
| India | S3 | Ram Maa Logistics Private Limited | G-1, Floor-Grd, One Avighna Park, Mahadeo Palav Marg, Curry Road Parel, Mumbai, Maharashtra | Ram Maa L0gisemtiarcs Private Limited | G-1, Mumbai, Mira-bhayandar, MH | 0.275 | 0.000 |
| India | S3 | Jaipur (india) Partners | 50, Sikar Road, Parasram Nagar, Dahar Ke Balaji, Jaipur, Rajasthan | Jaipur (india) | Jaipur, Door No 820 50, RJ | 0.000 | 0.000 |
| India | S3 | Seven Sun Producer Private Limited | Unit 508, Crescent Business Square, 5Th Floor, Opp Gundecha Onclave, Khairani Road, Andheri East, Mumbai, Maharashtra | Priavfe Seven Sun Producer Limited | MH, Mumbai Region, Unit 08, Mumbai | 0.000 | 0.000 |
| India | S3 | Mumbai Marketing Private Limited | Flat No.5, 2Nd Floor, Nalini Apartment, Plot No.347/A, Linking Road, Khar West, Mumbai, Mumbai City, Maharashtra | Mumbai Marketing Private Limited | H.no 5, Raigad, Mumbai City, MH | 1.000 | 0.000 |

### partial_name_and_address_noise

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| India | S3 | Prem & Sons Pvt Ltd | D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka | Prem & Pvt Ltd Partners | D-4, Bangalore, KA | 0.197 | 0.000 |
| India | S3 | Anushree Sangh | Row House No B4, 14 /15, Hill View, Surekha Niwas Siddhivinayak Nagari, Nigdi, Pune, Maharashtra | Anushree [Services] | Row House No B4, Pune, Pimpri Chinchwad, MH | 0.608 | 0.000 |
| India | S3 | Station Global Co | No. 309, Shreshtabhumi No. 87, K.R. Road. Shankarapuram, Bangalore, Karnataka | Station  Gall Co | #309, Bengaluru, KA | 0.522 | 0.000 |
| India | S3 | Lustre & Partners | Telangana, Sumitra Nagar, Baghameeri, H.No.5-5, Hyderabad, Tirumalagiri | Lustre + Palors | No 5, Hyderabad, తెలంగాణ | 0.519 | 0.000 |
| India | S3 | Mountain Mastermind Private Limited | A-57/1, Dlf 1, Gurugram, Dlf Qe, Gurgaon, Haryana | Mountain Private  Limited Center | Hn 30 A-57/1, Gurgaon, हरियाणा | 0.480 | 0.000 |

### name_replaced_address_changed

| country | source | s1_name | s1_addr | r_name | r_addr | name_cos | addr_cos |
|---|---|---|---|---|---|---|---|
| India | S3 | Seabird (India) Projects-Lucknow | Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh | Xylonexbrix | 3/7, Lucknow, UP | 0.000 | 0.402 |
| India | S2 | Aarvita Biosciences Private Limited | No.41, Ponnambalam Salai, K.K.Nagar, Chennai, Tamil Nadu | Jaxrizagild | 41, CHENNAI CITY REGION, தமிழ்நாடு | 0.000 | 0.000 |
| India | S3 | Global Traders Private Limited | 17/1, New Road, Jhansi, Uttar Pradesh | Mirahal0 | 17/1, Lalitpur, New Road, UP | 0.000 | 0.000 |
| India | S3 | Vision Impex Private Limited | Flat No - 003, B-23, R N B Bldg, Anand Nagar, Dahisar East, Mumbai, Maharashtra | Lumarc | Flat No - 003, Dahisar East, Mumbai, MH | 0.000 | 0.000 |
| US | S2 | Wood Integrated Oncology LLC | 2228 Sandra Terrace, Phoenix, AZ | Synavikor | 222 SANDRA TER, SUNNYSLOPE, AZ | 0.000 | 0.339 |
