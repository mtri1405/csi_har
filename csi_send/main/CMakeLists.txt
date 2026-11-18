idf_component_register(SRCS "main.c" "udp_sender.c" "wifi_ap.c"
                       INCLUDE_DIRS "." REQUIRES esp_wifi esp_event esp_netif nvs_flash lwip)