
from typing import TextIO;

import boto3;


def runMed(text: str, out: TextIO, region: str = "us-west-2") -> None:
    client = boto3.client(service_name='comprehendmedical', region_name=region)
    result = client.detect_entities(Text=text)
    entities = result['Entities'];
    out.write(f"{text}\n");
    for entity in entities:
        out.write(f"{entity}\n");


if __name__ == "__main__":
    runMed("estradurin 40mg injection (pdr for recon)+diluent");

