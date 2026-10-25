"""
Land eines Turniers (IOC-Code wie in Sackmanns *_players.csv) für das Heimvorteil-Feature.

Quellen der Reihe nach: ESPN-Venue ('Shanghai, China PR' -> Landesname), tennis-data-'Location'
(Stadt), dann Wörter im Turniernamen. Die Tabelle deckt die Orte der ATP- und WTA-Tour seit 2015
ab. Unbekannt -> '' (dann ist das Heim-Feature 0).
"""
import re
import unicodedata

# Stadt / Turnier-Stichwort -> IOC. Schlüssel normalisiert (klein, ohne Akzente/Leerzeichen).
CITY_IOC = {
    # Grand Slams und Finals
    'australianopen': 'AUS', 'melbourne': 'AUS', 'rolandgarros': 'FRA', 'frenchopen': 'FRA', 'paris': 'FRA',
    'wimbledon': 'GBR', 'london': 'GBR', 'usopen': 'USA', 'newyork': 'USA', 'flushingmeadows': 'USA',
    'turin': 'ITA', 'torino': 'ITA', 'nittoatpfinals': 'ITA', 'milan': 'ITA', 'jeddah': 'KSA', 'riyadh': 'KSA',
    'cancun': 'MEX', 'shenzhen': 'CHN', 'zhuhai': 'CHN', 'singapore': 'SGP', 'fortworth': 'USA',
    # Australien / Ozeanien
    'brisbane': 'AUS', 'sydney': 'AUS', 'adelaide': 'AUS', 'perth': 'AUS', 'hobart': 'AUS', 'auckland': 'NZL',
    # Asien
    'doha': 'QAT', 'dubai': 'UAE', 'abudhabi': 'UAE', 'beijing': 'CHN', 'shanghai': 'CHN', 'wuhan': 'CHN',
    'chengdu': 'CHN', 'hangzhou': 'CHN', 'ningbo': 'CHN', 'guangzhou': 'CHN', 'tianjin': 'CHN', 'nanchang': 'CHN',
    'jiujiang': 'CHN', 'zhengzhou': 'CHN', 'tokyo': 'JPN', 'osaka': 'JPN', 'hiroshima': 'JPN', 'seoul': 'KOR',
    'pune': 'IND', 'chennai': 'IND', 'mumbai': 'IND', 'bengaluru': 'IND', 'kualalumpur': 'MAS',
    'hongkong': 'HKG', 'taipei': 'TPE', 'astana': 'KAZ', 'nursultan': 'KAZ', 'almaty': 'KAZ', 'tashkent': 'UZB',
    'monastir': 'TUN', 'rabat': 'MAR', 'marrakech': 'MAR', 'cairo': 'EGY', 'telaviv': 'ISR',
    'antalya': 'TUR', 'istanbul': 'TUR', 'huahin': 'THA', 'manila': 'PHI',
    # Europa
    'rotterdam': 'NED', 'shertogenbosch': 'NED', 'denbosch': 'NED', 'rosmalen': 'NED', 'hertogenbosch': 'NED',
    'marseille': 'FRA', 'montpellier': 'FRA', 'lyon': 'FRA', 'metz': 'FRA', 'nice': 'FRA', 'strasbourg': 'FRA',
    'rouen': 'FRA', 'limoges': 'FRA', 'bercy': 'FRA',
    'madrid': 'ESP', 'barcelona': 'ESP', 'mallorca': 'ESP', 'marbella': 'ESP', 'valencia': 'ESP',
    'rome': 'ITA', 'roma': 'ITA', 'parma': 'ITA', 'palermo': 'ITA', 'cagliari': 'ITA', 'florence': 'ITA',
    'firenze': 'ITA', 'naples': 'ITA', 'sardinia': 'ITA',
    'hamburg': 'GER', 'halle': 'GER', 'munich': 'GER', 'stuttgart': 'GER', 'berlin': 'GER', 'cologne': 'GER',
    'koln': 'GER', 'badhomburg': 'GER', 'nurnberg': 'GER', 'nuremberg': 'GER',
    'vienna': 'AUT', 'wien': 'AUT', 'kitzbuhel': 'AUT', 'linz': 'AUT', 'graz': 'AUT',
    'basel': 'SUI', 'geneva': 'SUI', 'gstaad': 'SUI', 'lausanne': 'SUI', 'biel': 'SUI',
    'montecarlo': 'MON', 'monaco': 'MON',
    'eastbourne': 'GBR', 'queens': 'GBR', 'birmingham': 'GBR', 'nottingham': 'GBR', 'edgbaston': 'GBR',
    'antwerp': 'BEL', 'brussels': 'BEL', 'umag': 'CRO', 'zagreb': 'CRO', 'bastad': 'SWE',
    'stockholm': 'SWE', 'bucharest': 'ROU', 'clujnapoca': 'ROU', 'cluj': 'ROU', 'iasi': 'ROU',
    'sofia': 'BUL', 'belgrade': 'SRB', 'banjaluka': 'BIH', 'budapest': 'HUN', 'gdynia': 'POL', 'warsaw': 'POL',
    'prague': 'CZE', 'ostrava': 'CZE', 'portoroz': 'SLO', 'lisbon': 'POR', 'estoril': 'POR', 'oeiras': 'POR',
    'helsinki': 'FIN', 'moscow': 'RUS', 'stpetersburg': 'RUS', 'saintpetersburg': 'RUS', 'kremlin': 'RUS',
    'tallinn': 'EST', 'athens': 'GRE', 'luxembourg': 'LUX', 'bol': 'CRO', 'gijon': 'ESP', 'dublin': 'IRL',
    # Amerika
    'indianwells': 'USA', 'miami': 'USA', 'cincinnati': 'USA', 'mason': 'USA', 'washington': 'USA',
    'atlanta': 'USA', 'winstonsalem': 'USA', 'delraybeach': 'USA', 'houston': 'USA', 'dallas': 'USA',
    'sandiego': 'USA', 'charleston': 'USA', 'newport': 'USA', 'losangeles': 'USA', 'stanford': 'USA',
    'sanjose': 'USA', 'memphis': 'USA', 'newhaven': 'USA', 'cleveland': 'USA', 'chicago': 'USA',
    'austin': 'USA', 'charlotte': 'USA', 'columbus': 'USA', 'orlando': 'USA', 'tampa': 'USA',
    'montreal': 'CAN', 'toronto': 'CAN', 'quebec': 'CAN', 'quebeccity': 'CAN', 'granby': 'CAN',
    'acapulco': 'MEX', 'loscabos': 'MEX', 'monterrey': 'MEX', 'guadalajara': 'MEX',
    'merida': 'MEX', 'bogota': 'COL', 'buenosaires': 'ARG', 'cordoba': 'ARG', 'riodejaneiro': 'BRA',
    'saopaulo': 'BRA', 'florianopolis': 'BRA', 'santiago': 'CHI', 'quito': 'ECU', 'lima': 'PER',
    # weitere Orte (v.a. WTA)
    'casablanca': 'MAR', 'katowice': 'POL', 'lugano': 'SUI', 'pazardzhik': 'BUL', 'phillipisland': 'AUS',
    'yarravalley': 'AUS', 'gippsland': 'AUS', 'pattaya': 'THA', 'badgastein': 'AUT', 'baku': 'AZE',
    'kaohsiung': 'TPE', 'jurmala': 'LAT', 'courmayeur': 'ITA', 'tenerife': 'ESP', 'jiangxi': 'CHN',
    'saintmalo': 'FRA', 'concord': 'USA', 'midland': 'USA', 'newhaven': 'USA', 'hongkongopen': 'HKG',
    'bogotawta': 'COL', 'karlsruhe': 'GER', 'contrexeville': 'FRA', 'monastir': 'TUN', 'dobrich': 'BUL',
    'sanjuan': 'PUR', 'nanjing': 'CHN', 'kunming': 'CHN', 'anning': 'CHN', 'calvia': 'ESP',
    'lexington': 'USA', 'ljubljana': 'SLO', 'montreux': 'SUI', 'brescia': 'ITA', 'ilkley': 'GBR',
    'makarska': 'CRO', 'vancouver': 'CAN', 'vanopen': 'CAN', 'philly': 'USA', 'philadelphia': 'USA',
    'macon': 'FRA', 'leipzig': 'GER', 'koksijde': 'BEL', 'figueiradafoz': 'POR', 'caldasdarainha': 'POR',
    'puglie': 'ITA', 'bari': 'ITA', 'tolentino': 'ITA', 'tauste': 'ESP', 'chicagowomens': 'USA',
}

# Landesnamen (ESPN-Venue) -> IOC
COUNTRY_IOC = {
    'australia': 'AUS', 'france': 'FRA', 'unitedkingdom': 'GBR', 'england': 'GBR', 'greatbritain': 'GBR',
    'unitedstates': 'USA', 'usa': 'USA', 'italy': 'ITA', 'spain': 'ESP', 'germany': 'GER', 'austria': 'AUT',
    'switzerland': 'SUI', 'netherlands': 'NED', 'belgium': 'BEL', 'sweden': 'SWE', 'croatia': 'CRO',
    'serbia': 'SRB', 'hungary': 'HUN', 'poland': 'POL', 'czechrepublic': 'CZE', 'czechia': 'CZE',
    'portugal': 'POR', 'finland': 'FIN', 'russia': 'RUS', 'romania': 'ROU', 'bulgaria': 'BUL',
    'monaco': 'MON', 'greece': 'GRE', 'slovenia': 'SLO', 'estonia': 'EST', 'luxembourg': 'LUX',
    'bosniaandherzegovina': 'BIH', 'canada': 'CAN', 'mexico': 'MEX', 'argentina': 'ARG', 'brazil': 'BRA',
    'chile': 'CHI', 'colombia': 'COL', 'ecuador': 'ECU', 'peru': 'PER', 'china': 'CHN', 'chinapr': 'CHN',
    'japan': 'JPN', 'korea': 'KOR', 'southkorea': 'KOR', 'korearepublic': 'KOR', 'india': 'IND',
    'qatar': 'QAT', 'unitedarabemirates': 'UAE', 'uae': 'UAE', 'saudiarabia': 'KSA', 'kazakhstan': 'KAZ',
    'uzbekistan': 'UZB', 'hongkong': 'HKG', 'hongkongchina': 'HKG', 'chinesetaipei': 'TPE', 'taiwan': 'TPE',
    'thailand': 'THA', 'malaysia': 'MAS', 'singapore': 'SGP', 'turkey': 'TUR', 'turkiye': 'TUR',
    'tunisia': 'TUN', 'morocco': 'MAR', 'egypt': 'EGY', 'israel': 'ISR', 'newzealand': 'NZL',
    'ireland': 'IRL', 'philippines': 'PHI',
}


def _norm(text):
    s = unicodedata.normalize("NFKD", str(text or ''))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z ]", "", s)


def _compact(text):
    return _norm(text).replace(" ", "")


def tourney_country(name='', location='', country=''):
    """IOC-Code des Turnierorts oder ''."""
    if country and _compact(country) in COUNTRY_IOC:
        return COUNTRY_IOC[_compact(country)]
    if location and _compact(location) in CITY_IOC:
        return CITY_IOC[_compact(location)]
    words = _norm(name).split()
    # Zweier-Kombinationen zuerst ('Indian Wells', 'Monte Carlo', 'Us Open'), dann einzelne Wörter
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            key = "".join(words[i:i + n])
            if key in CITY_IOC:
                return CITY_IOC[key]
    return ''
