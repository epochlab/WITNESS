#!/usr/bin/env python3

import re, pwd, datetime, stat, mimetypes
from pathlib import Path

import exifread

def file_metadata(file_path: Path) -> dict:
    file_stat = file_path.stat()
    return {
        "filename": file_path.name,
        "folder": file_path.parent.absolute(),
        "file_type": mimetypes.guess_type(file_path)[0].split('/')[1].upper(),
        "file_type_extension": file_path.suffix.replace('.', ''),
        "mime_type": mimetypes.guess_type(file_path)[0],
        "filesize": format_bytes(file_stat.st_size),
        "owner": pwd.getpwuid(file_stat.st_uid).pw_name,
        "permissions": stat.filemode(file_stat.st_mode),
        "last_access": datetime.datetime.fromtimestamp(file_stat.st_atime),
        "last_modified": datetime.datetime.fromtimestamp(file_stat.st_mtime),
        "metadata_changed": datetime.datetime.fromtimestamp(file_stat.st_ctime),
        "birth_time": datetime.datetime.fromtimestamp(file_stat.st_birthtime) if hasattr(file_stat, 'st_birthtime') else "Not available",
        "ballistics": ballistics(file_path.name),
    }

def exif_data(file_path: Path) -> dict:
    exif_metadata = {}
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f)
    excluded_tags = ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']
    for tag, value in tags.items():
        if tag not in excluded_tags and str(value).strip():
            group, _, description = tag.partition(' ')
            exif_metadata[f"{group}_{description}"] = str(value)
    sorted_metadata = dict(sorted(exif_metadata.items(), key=lambda item: item[0].split('_', 1)[0]))
    return sorted_metadata

def ballistics(file_path: str) -> str:
    BALLISTICS_TABLE = [
        (re.compile("^DSCN[0-9]{4}\\.JPG$", re.IGNORECASE), "Nikon Coolpix camera"),
        (re.compile("^DSC_[0-9]{4}\\.JPG$", re.IGNORECASE), "Nikon digital camera"),
        (re.compile("^FUJI[0-9]{4}\\.JPG$", re.IGNORECASE), "Fujifilm digital camera"),
        (re.compile("^IMG_[0-9]{4}\\.JPG$", re.IGNORECASE), "Canon DSLR or iPhone camera"),
        (re.compile("^PIC[0-9]{5}\\.JPG$", re.IGNORECASE), "Olympus D-600L camera"),
    ]
    
    for pattern, desc in BALLISTICS_TABLE:
        if pattern.match(file_path):
            return desc
    return "Unknown source or manually renamed"

def format_bytes(bytes_size: float) -> str:
    if bytes_size < 1024: return f"{bytes_size} bytes"
    elif bytes_size < 1024**2: return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024**3: return f"{bytes_size / 1024**2:.2f} MB"
    else: return f"{bytes_size / 1024**3:.2f} GB"