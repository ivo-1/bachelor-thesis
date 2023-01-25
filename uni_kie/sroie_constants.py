from uni_kie import PATH_DATA

PATH_SROIE = PATH_DATA / "sroie"

PATH_SROIE_TEST = PATH_SROIE / "test" / "input"  # a folder with images
PATH_SROIE_TEST_PREDICTIONS = PATH_SROIE_TEST / "predictions"
PATH_SROIE_TEST_OCR = (
    PATH_SROIE / "test" / "ocr_boxes"
)  # a folder with txt files that contain the OCR boxes


class SROIE_CONSTANTS:
    prompt_key_to_gold_key = {
        "Company Name": "company",
        "Date of Receipt": "date",
        "Address of Company": "address",
        "Total": "total",
    }
    gold_key_to_prompt_key = {v: k for k, v in prompt_key_to_gold_key.items()}
    gold_keys = list(gold_key_to_prompt_key.keys())
    prompt_keys = list(prompt_key_to_gold_key.keys())

    SHOTS = [
        {
            "input": "SYARIKAT PERNIAGAAN GIN KEE\n(81109-A)\nNO 290, JALAN AIR PANAS.\nSETAPAK.\n53200, KUALA LUMPUR.\nTEL :03-40210276\nGST ID : 000750673920\nSIMPLIFIED TAX INVOICE\nCASH\nDOC NO.\n: CS00012504\nDATE:\n02/01/2018\nCASHIER\n: USER\nTIME:\n14:49:00\nSALESPERSON :\nREF.:\nITEM\nQTY\nS/PRICE\nAMOUNT\nTAX\n1762\n1\n7.95\n7.95\nSR\n17MM COMB SPANNER\n1041\n2\n15.90\n31.80\nSR\n6' X 35# CORRUGATED ROOFING SHEET\nTOTAL QTY:\n3\n39.75\nTOTAL SALES (EXCLUDING GST) :\n37.50\nDISCOUNT :\n0.00\nTOTAL GST :\n2.25\nROUNDING :\n0.00\nTOTAL SALES (INCLUSIVE OF GST) :\n39.75\nCASH :\n39.75\nCHANGE :\n0.00\nGST SUMMARY\nTAX CODE\nSR\n%\n6\nAMT (RM)\n37.50\nTAX (RM)\n2.25\nTOTAL :\n37.50\n2.25\nGOODS SOLD ARE NOTRETURNABLE. THANK YOU",
            "solution": {
                "company": "SYARIKAT PERNIAGAAN GIN KEE",
                "date": "02/01/2018",
                "address": "NO 290, JALAN AIR PANAS, SETAPAK, 53200, KUALA LUMPUR.",
                "total": "39.75",
            },
            "target_model_output": " SYARIKAT PERNIAGAAN GIN KEE\nDate of Receipt: 02/01/2018\nAddress of Company: NO 290, JALAN AIR PANAS, SETAPAK, 53200, KUALA LUMPUR.\nTotal: 39.75\n<|stop key|>",
        },
        {
            "input": "POPULAR BOOK\nCO. (M) SDN BHD\n(CO. NO. 113825-W)\n(GST REG NO. 001492992000)\nNO 8, JALAN 7/118B, DESA TUN RAZAK\n56000 KUALA LUMPUR, MALAYSIA\nAEON SHAH ALAM\nTEL : 03-55235214\n27/02/18 21:22\nTASHA\nSLIP NO. : 8020188757\nTRANS : 204002\nMEMBER CARD NO : 1001016668849\nCARD EXPIRY : 31/05/18\nDESCRIPTION\nAMOUNT\nDOCUMENT HOL A4 1466A-TRA\n2PC @ 1.15\nMEMBER DISCOUNT\nPB PVC A4 L-FLD PBA4L25\n2PC @ 3.90\nMEMBER DISCOUNT\nCANON CAL AS120V GREY\nMEMBER DISCOUNT\nNASI'APR16/SEASHORE\n[BK]\n2.30 T\n-0.24\n7.80 T\n-0.78\n29.90 T\n-2.99\n5.00 Z\nTOTAL RM INCL OF GST\nROUNDING ADJ\nTOTAL RM\nCASH\nCHANGE\n40.99\n0.01\n41.00\n-51.00\n10.00\nITEM COUNT\nGST SUMMARY\nT @ 6%\nZ @ 0%\nAMOUNT (RM)\n33.95\n5.00\n6\nTAX (RM)\n2.04\n0.00\nTOTAL SAVINGS\n-4.01\nBE A POPULAR CARD MEMBER\nAND ENJOY SPECIAL DISCOUNTS\nTHANK YOU. PLEASE COME AGAIN.\nWWW.POPULAR.COM.MY\nBUY CHINESE BOOKS ONLINE\nWWW.POPULARONLINE.COM.MY",
            "solution": {
                "company": "POPULAR BOOK CO. (M) SDN BHD",
                "date": "27/02/18",
                "address": "NO 8, JALAN 7/118B, DESA TUN RAZAK 56000 KUALA LUMPUR, MALAYSIA",
                "total": "41.00",
            },
            "target_model_output": "POPULAR BOOK CO. (M) SDN BHD\nDate of Receipt: 27/02/18\nAddress of Company: NO 8, JALAN 7/118B, DESA TUN RAZAK 56000 KUALA LUMPUR, MALAYSIA\nTotal: 41.00\n<|stop key|>",
        },
    ]
