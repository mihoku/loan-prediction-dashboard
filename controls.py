# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:00:57 2020

@author: joser
"""


#controls for web app

verdicts = [
    ["[True]","diprediksikan akan mengalami lebih dari dua kali amandemen date drawing limit"],
    ["[False]","diprediksikan tidak akan mengalami lebih dari dua kali amandemen date drawing limit"],
    ["DISBURSEMENT UNDER 50%","diprediksikan akan memiliki tingkat pencairan/realisasi yang rendah yakni di bawah 50% dari nilai komitmen"],
    ["DISBURSEMENT UNDER 90%","diprediksikan akan memiliki tingkat pencairan/realisasi antara 50-90% dari nilai komitmen"],
    ["FULLY DISBURSED","diprediksikan akan memiliki tingkat pencairan/kinerja realisasi yang baik, yakni di atas 90% nilai komitmen pinjaman dapat direalisasikan"]
    ]

LOANTYPE=["Pinjaman Luar Negeri","Pinjaman Dalam Negeri"]

PROGRAMS=["Pembelian Barang","Program Pemerintah","Pembangunan Infrastruktur"]

LENDERTYPE=["Kreditor Swasta Asing (KSA)","Lembaga Penjamin Kredit Ekspor (LPKE)","Lembaga Multilateral","Negara (Bilateral)","Bank BUMN Dalam Negeri","Bank Swasta Dalam Negeri"]

EATYPE=["Kementerian/Lembaga","Badan Usaha Milik Negara","Pemerintah Daerah"]

TARGET1={
    "0":"Numerous Drawing Limit Amendment: False",
    "1": "Numerous Drawing Limit Amendment: True"
    }

TARGET2={
    "0":"Disbursement Under 50%",
    "1":"Disbursement Under 90%",
    "2":"Fully Disbursed"
    }