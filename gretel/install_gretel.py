from gretel_client import get_cloud_client

client = get_cloud_client(prefix="api", api_key="grtu9e24d8a5c790b1bde52b051da71a8a271f582ba1992d4eb9de839aa916af0577")
client.install_packages()
