idf_component_register(SRCS "csi_recv.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_timer esp_driver_uart csi_proto)