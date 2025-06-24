import requests
import pandas as pd
import gfwapiclient as gfw
import os
import asyncio
from shapely.geometry import box, mapping


access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbW" \
"UiOiJXaWxsIFBvbmN6YWsiLCJ1c2VySWQiOjQ2NzE1LCJhcHBsaWNhdGlvbk5hbWUiOiJXaWxsIFBvbmN6YWsi" \
"LCJpZCI6MjgwNywidHlwZSI6InVzZXItYXBwbGljYXRpb24ifSwiaWF0IjoxNzQ4OTU5MjIwLCJleHAiOjIwN" \
"jQzMTkyMjAsImF1ZCI6ImdmdyIsImlzcyI6ImdmdyJ9.i9JoOMrqnRuQks9RwcBJpbohYXvby7UigXgImksE6pD" \
"6FVZXuC-2-rZBoPCxNpQTJAtVDtgf5VunVxvs0Eelry88PyLRtzeXqRrE4YjH_wVGPuk4ZrXz5c9a4PnTt8xbglq" \
"ohIXha403c37UCDIPeV4Z7y8e3e6WxCWkaiW-5kD0Z-23CIB9ZC7gWh01fL64HqaQruCOIyRD_hRJB1PUdy0t9J" \
"dwk56lqcipcjvQVLZXvInzk-twhzIefaWZuLLBbdSOfWtlt6-NBqdaidCQfe6QChoHDa3lFUNMHCcQMOe_geYsj" \
"ObvXvuX6R0_4CmWq7E4t6w52_y0SjLKvVTScyz95RBm7xQMjmrFsx4VsHwpLNv9i_DtUKQuVt_OHGA6DwgIdtV0r" \
"sOKyPYosH4OLELMNjPuEPi9VC6HR-lagVTMnsGNCofsoOzPlbJNVQkaE2K2hFnD9rWAJF1F1efj9PmPznsahB9Jo" \
"km0Mw8lOiU74EkKFtYKIsXam2E2QOs3"

gfw_client = gfw.Client(access_token=access_token)

geometry = {
    "type": "Polygon",
    "coordinates": [
        [
            [-180.0, -85.0511287798066],
            [-180.0, 0.0],
            [0.0, 0.0],
            [0.0, -85.0511287798066],
            [-180.0, -85.0511287798066],
        ]
    ],
}

async def main():
   
    sar_infrastructure_result = await gfw_client.datasets.get_sar_fixed_infrastructure(
        geometry=geometry
    )

    sar_infrastructure_df = sar_infrastructure_result.df()
    print(sar_infrastructure_df.info())
    print(sar_infrastructure_df.head())

if __name__ == "__main__":
    asyncio.run(main())

