import re

long_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
short_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def __str2month(dstr):
    for mi in range(12):
        if long_months[mi].lower() in dstr.lower() or short_months[mi].lower() in dstr.lower():
            return mi + 1
    return None


def __is_int(teststr):
    try:
        int(teststr)
    except ValueError:
        return False
    return True


def __makes_sense(dt):
    year = 1800 < dt['year'] < 2100
    month = True
    if dt['month'] is not None:
        month = 0 < dt['month'] <= 12
    day = True
    if dt['day'] is not None:
        day = 0 < dt['day'] <= 31

    return year and month and day


def parse_date(dstr):
    ret = {
        "year": None,
        "month": None,
        "day": None
    }
    try:
        if re.search(r'\d{1,2}[^0-9a-z:]+\d{1,2}[^0-9a-z:]+\d{4}', dstr, flags=re.IGNORECASE) is not None:
            matches = re.search(r'(\d{1,2})[^0-9a-z:]+(\d{1,2})[^0-9a-z:]+(\d{4})', dstr, flags=re.IGNORECASE)
            m1 = int(matches.group(1)) if __is_int(matches.group(1)) else 0
            m2 = int(matches.group(2)) if __is_int(matches.group(2)) else 0
            m3 = int(matches.group(3)) if __is_int(matches.group(3)) else 0
            tmp = {
                "year": m3
            }
            if m1 <= 12 and m2 >= 12:
                tmp.update({
                    "month": m1,
                    "day": m2
                })
            elif m1 >= 12 and m2 <= 12 or m1 <= 12 and m2 <= 12:
                tmp.update({
                    "month": m2,
                    "day": m1
                })

            if __makes_sense(tmp):
                ret.update(tmp)
                return ret

        if re.search(r'\d{4}[^0-9a-z:]+\d{1,2}[^0-9a-z:]+\d{1,2}', dstr, flags=re.IGNORECASE) is not None:
            matches = re.search(r'(\d{4})[^0-9a-z:]+(\d{1,2})[^0-9a-z:]+(\d{1,2})', dstr, flags=re.IGNORECASE)
            m1 = int(matches.group(1)) if __is_int(matches.group(1)) else 0
            m2 = int(matches.group(2)) if __is_int(matches.group(2)) else 0
            m3 = int(matches.group(3)) if __is_int(matches.group(3)) else 0
            tmp = {
                "year": m1
            }
            if m3 <= 12 and m2 >= 12:
                tmp.update({
                    "month": m3,
                    "day": m2
                })
            elif m3 >= 12 and m2 <= 12 or m3 <= 12 and m2 <= 12:
                tmp.update({
                    "month": m2,
                    "day": m3
                })
            if __makes_sense(tmp):
                ret.update(tmp)
                return ret

        if re.search(r'\w+[^0-9a-z:]+\d{1,2}[^0-9a-z:]+\d{4}', dstr, flags=re.IGNORECASE) is not None:
            matches = re.search(r'(\w+)[^0-9a-z:]+?(\d{1,2})[^0-9a-z:]+(\d{4})', dstr, flags=re.IGNORECASE)

            m1 = __str2month(dstr)
            m2 = int(matches.group(2)) if __is_int(matches.group(2)) else 0
            m3 = int(matches.group(3)) if __is_int(matches.group(3)) else 0

            tmp = {
                "year": m3,
                "month": m1,
                "day": m2
            }
            if m1 is not None and __makes_sense(tmp):
                ret.update(tmp)
                return ret

        if re.search(r'\d{1,2}[^0-9a-z:]+\w+[^0-9a-z:]+\d{4}', dstr, re.IGNORECASE) is not None:
            matches = re.search(r'(\d{1,2})[^0-9a-z:]+(\w+)[^0-9a-z:]+(\d{4})', dstr, re.IGNORECASE)
            m1 = int(matches.group(1)) if __is_int(matches.group(1)) else 0
            m2 = __str2month(dstr)
            m3 = int(matches.group(3)) if __is_int(matches.group(3)) else 0

            tmp = {
                "year": m3,
                "month": m2,
                "day": m1
            }
            if m2 is not None and __makes_sense(tmp):
                ret.update(tmp)
                return ret

        ret.update({
            "year": re.search('(\d{4})', dstr).group(1)
        })
        ret.update({
            "month": __str2month(dstr)
        })

    except:
        pass
    finally:
        return ret
