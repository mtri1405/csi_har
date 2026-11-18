idf_component_register(SRCS "csi_receiver.c" "wifi_sta.c" "main.c"
                    INCLUDE_DIRS "." REQUIRES esp_wifi esp_event esp_netif nvs_flash lwip)