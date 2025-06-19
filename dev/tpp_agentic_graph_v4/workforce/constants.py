"""
Constants to be used in all this proyect
"""

# %%
from typing import Literal

# region Proyect metadata
APP_TITLE: str = "Capacity_IA"
APP_VERSION: str = "v0.2.0"
APP_DESCRIPTION: str = "Herramientas de Capacity, un motor para generar forecasts, configuraciones de atención, y recomendaciones."
# endregion Proyect metadata

# region Hardcoded connection string
# Yeah, I also cringe with this. Bite me.
DB_SERVER = "ttp-dev-sql.database.windows.net"
DB_DATABASE = "ttp-dev-capacity"
DB_USERNAME = "ttp_dev_usr_capacity"
DB_PASSWORD = "qrRtgtagbb28164"
# endregion Hardcoded connection string

# region Decorative thingies

CAPACITY_LOGO: str = """

   _____                            _  _           _____             
  / ____|                          (_)| |         |_   _|    /\      
 | |      __ _  _ __    __ _   ___  _ | |_  _   _   | |     /  \     
 | |     / _` || '_ \  / _` | / __|| || __|| | | |  | |    / /\ \    
 | |____| (_| || |_) || (_| || (__ | || |_ | |_| | _| |_  / ____ \   
  \_____|\__,_|| .__/  \__,_| \___||_| \__| \__, ||_____|/_/    \_\  
               | |                           __/ |                   
               |_|                          |___/  """

# endregion Decorative thingies

# region Actual useful thingies
ServiceConfiguration = Literal["Fifo", "FIFO", "Alternancia", "Rebalse"]
"""
Service mode for a Desk

- First-In, First-Out (FIFO)
- Alternancia (this n times, then that nn times, repeat) [round-robin-like]
- Rebalse (these, and those if you are done with these) [priority]
"""

UUID_ZERO: str = "00000000-0000-0000-0000-000000000000"
"""Represents the UUID zero, as a string."""

# endregion Actual useful thingies
