""" Splich.py
Splits files into parts, or in chunk_size
Splich is a file splitting tool that allows you to split a file into parts, and reassembles them

https://github.com/shine-jayakumar/splich

Author: Shine Jayakumar
https://github.com/shine-jayakumar

MIT License

Copyright (c) 2022 Shine Jayakumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from io import BytesIO
import os
import glob
import hashlib
from datetime import datetime

VERSION = "v.1.4"

VERBOSE = False


def file_split(file, parts=None, chunk_size=None):
    """
    Splits files into parts, or in chunk_size
    """
    if not file:
        return False
    if not parts and not chunk_size:
        return False

    fsize = os.path.getsize(file)

    if chunk_size and chunk_size > fsize:
        raise ValueError("Chunk size cannot be greater than file size")

    vvprint(f"Source file: {file}")
    vvprint(f"Size: {fsize}")

    segment_size = 0

    if parts:
        segment_size = fsize // parts
    else:
        segment_size = chunk_size

    if segment_size < 1:
        raise ValueError("At least 1 byte required per part")

    vvprint(f"Segment Size: {segment_size}")

    fdir, fname = os.path.split(file)
    # fname = fname.split('.')[0]
    fname = os.path.splitext(fname)[0]

    vvprint("Generating hash")
    file_hash = gethash(file)
    start_time = datetime.today().strftime("%m%d%Y_%H%M")

    vvprint(f"Hash: {file_hash}\n\n")
    vvprint(f"Reading file: {file}")

    with open(file, "rb") as fh:
        fpart = 1
        while fh.tell() != fsize:
            if parts:
                # check if this is the last part
                if fpart == parts:
                    # size of the file - wherever the file pointer is
                    # the last part would contain segment_size + whatever is left of the file
                    segment_size = fsize - fh.tell()

            chunk = fh.read(segment_size)
            part_filename = os.path.join(fdir, f"{fname}_{start_time}_{fpart}.prt")
            vvprint(f"{part_filename} Segment size: {segment_size} bytes")
            with open(part_filename, "wb") as chunk_fh:
                chunk_fh.write(chunk)
            fpart += 1

        # hashfile generation
        hashfilename = f"{fname}_hash_{start_time}"
        hashfile_path = os.path.join(fdir, hashfilename)
        vvprint(f"Hashfile: {hashfile_path}")
        with open(hashfile_path, "w", encoding="utf-8") as hashfile:
            hashfile.write(file_hash)

        return True


def file_stitch(file, outfile=None):
    """
    Stitches the parts together
    """
    # d:\\somedir\\somefile.txt to
    # d:\\somedir and somefile.txt

    if not file:
        return False

    fdir, fname = os.path.split(file)
    # fname = fname.split('.')[0]
    fname = os.path.splitext(fname)[0]

    file_parts = glob.glob(os.path.join(fdir, f"{fname}_*.prt"))
    file_parts = sort_file_parts(file_parts)

    if not file_parts:
        print(f"Split File Path: {file}")
        raise FileNotFoundError("Split files not found")

    #    if outfile:
    #        # if just the filename
    #        if os.path.split(outfile)[0] == '':
    #            # create the file in input dir (fdir)
    #            outfile = os.path.join(fdir, outfile)

    vvprint(f"Output: {outfile or file}")

    buffer = BytesIO()
    with open(outfile or file, "wb"):
        for filename in file_parts:
            vvprint(f"Reading {filename}")
            with open(filename, "rb") as prt_fh:
                buffer.write(prt_fh.read())

    vvprint(f"Written {os.path.getsize(outfile or file)} bytes")
    return buffer


def gethash(file):
    """
    Returns the hash of file
    """
    file_hash = None
    with open(file, "rb") as fh:
        file_hash = hashlib.sha256(fh.read()).hexdigest()
    return file_hash


def checkhash(file, hashfile):
    """
    Compares hash of a file with original hash read from a file
    """
    curhash = None
    orghash = None
    curhash = gethash(file)
    with open(hashfile, "r", encoding="utf-8") as fh:
        orghash = fh.read()

    return curhash == orghash


def vvprint(text):
    """
    print function to function only when verbose mode is on
    """
    if VERBOSE:
        print(text)


def getpartno(filepart):
    """
    Returns the part number from a part filename
    Ex: flask_05112022_1048_3.prt -> 3
    """
    return int(filepart.split("_")[-1].split(".")[0])


def sort_file_parts(file_part_list):
    """
    Returns a sorted list of part filenames based on the part number
    Ex: ['flask_05112022_1048_3.prt', 'flask_05112022_1048_1.prt', 'flask_05112022_1048_2.prt'] ->
        ['flask_05112022_1048_1.prt', 'flask_05112022_1048_2.prt', 'flask_05112022_1048_3.prt']
    """
    # creates list of (prt_no, part)
    fparts = [(getpartno(prt), prt) for prt in file_part_list]
    fparts.sort(key=lambda x: x[0])
    fparts = [prt[1] for prt in fparts]
    return fparts


def file_stitch_buffer(file):
    """
    Stitches the parts together and return BytesIO object
    """
    # d:\\somedir\\somefile.txt to
    # d:\\somedir and somefile.txt

    if not file:
        return False

    fdir, fname = os.path.split(file)
    # fname = fname.split('.')[0]
    fname = os.path.splitext(fname)[0]

    file_parts = glob.glob(os.path.join(fdir, f"{fname}_*.prt"))
    file_parts = sort_file_parts(file_parts)

    if not file_parts:
        raise FileNotFoundError("Split files not found")

    #    if outfile:
    #        # if just the filename
    #        if os.path.split(outfile)[0] == '':
    #            # create the file in input dir (fdir)
    #            outfile = os.path.join(fdir, outfile)

    buffer = BytesIO()
    for filename in file_parts:
        with open(filename, "rb") as prt_fh:
            buffer.write(prt_fh.read())

    print("Read in byte array")
    return buffer
