from pathlib import Path
from datetime import datetime, timedelta


def _parse_hmi_time(fname):
    parts = fname.stem.split('.')
    date_part = parts[2] + '.' + parts[3] + '.' + parts[4]
    dt_str = date_part.replace('_TAI', '').replace('.', '-', 2).replace('_', ' ', 1).replace('_', ':')
    return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') 

def find_closest_aia(time, aia_maps):
    from astropy.time import Time
    target = Time(datetime.fromisoformat(time))
    return min(aia_maps, key=lambda m: abs((m.date - target).to('s').value))

def find_closest_hmi(time, hmi_dir):
    target = datetime.fromisoformat(time) + timedelta(minutes=1) # HMI data is tagged TAI, which is about 37 seconds ahead of UTC, so we add a minute to be safe
    fits_files = list(Path(hmi_dir).glob('*.fits'))
    return min(fits_files, key=lambda f: abs(_parse_hmi_time(f) - target))