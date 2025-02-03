
import os
import sys
from typing import TextIO;
from time import time_ns;

import boto3;


def runMed(text: str, out: TextIO | None = None, region: str = "us-west-2") -> None:
    client = boto3.client(service_name='comprehendmedical', region_name=region)
    result = client.detect_entities(Text=text)
    entities = result['Entities'];
    tarF: TextIO = out if out is not None else open("/dev/stdout", "w");
    tarF.write(f"{text}\n");
    for entity in entities:
        tarF.write(f"{entity}\n");
    if out is None:
        tarF.close();


def getICD(text: str, out: str = "/dev/stdout", region: str = "us-west-2") -> None:
    client = boto3.client(service_name='comprehendmedical', region_name=region)
    result = client.infer_icd10_cm(Text=text)
    entities = result['Entities'];
    tarF: TextIO = open(out, "w");
    # tarF.write(f"{text}\n");
    for entity in entities:
        tarF.write(f"{entity}\n");
    tarF.close();


if __name__ == "__main__":
    # runMed("estradurin 40mg injection (pdr for recon)+diluent");
    getICD("Estradiol is indicated in various preparations for the treatment of moderate to severe vasomotor symptoms and vulvar and vaginal atrophy due to menopause, for the treatment of hypoestrogenism due to hypogonadism, castration, or primary ovarian failure, and for the prevention of postmenopausal osteoporosis. It is also used for the treatment of breast cancer (only for palliation therapy) in certain men or women with metastatic disease, and for the treatment of androgen-dependent prostate cancer (only for palliation therapy).27,30,31 It is also used in combination with other hormones as a component of oral contraceptive pills for preventing pregnancy (most commonly as Ethinylestradiol, a synthetic form of estradiol). A note on duration of treatment. Recommendations for treatment of menopausal symptoms changed drastically following the release of results and early termination of the Women's Health Initiative (WHI) studies in 2002 as concerns were raised regarding estrogen use.11 Specifically, the combined estrogen–progestin group was discontinued after about 5 years of follow up due to a statistically significant increase in invasive breast cancer and in cardiovascular events.12 Following extensive critique of the WHI results, Hormone Replacement Therapy (HRT) is now recommended to be used only for a short period (for 3-5 years postmenopause) in low doses, and in women without a history of breast cancer or increased risk of cardiovascular or thromboembolic disease.13 Estrogen for postmenopausal symptoms should always be given with a progestin component due to estrogen's stimulatory effects on the endometrium; in women with an intact uterus, unopposed estrogen has been shown to promote the growth of the endometrium which can lead to endometrial hyperplasia and possibly cancer over the long-term.")

