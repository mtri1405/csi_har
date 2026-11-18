#include "wifi_sta.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include "freertos/task.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"

/* Network name and password of the dedicated sensing network.
 * Change both before flashing and keep them identical on AP and STA.
 * WPA2 needs a password of at least 8 characters. */
#define WIFI_SSID       "CSI_SENSOR_AP"
#define WIFI_PASS       "12345678"

#define RX_IP_A         192
#define RX_IP_B         168
#define RX_IP_C         4
#define RX_IP_D         2       // RX1 = 192.168.4.2

#define GW_IP_D         1       // TX/AP = 192.168.4.1

static const char *TAG = "WIFI_STA";
static EventGroupHandle_t wifi_event_group;

#define WIFI_CONNECTED_BIT BIT0

static void event_handler(void *arg,
                          esp_event_base_t event_base,
                          int32_t event_id,
                          void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Connecting to AP...");
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected. Reconnecting...");
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;

        ESP_LOGI(TAG, "====================================");
        ESP_LOGI(TAG, "Connected Successfully!");
        ESP_LOGI(TAG, "IP Address : " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "====================================");

        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// Drain UDP payloads: CSI itself is delivered independently by the PHY callback.
static void udp_sink_task(void *arg)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons(3333);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (sock < 0 || bind(sock, (struct sockaddr *)&address, sizeof(address)) < 0) {
        ESP_LOGE(TAG, "Cannot bind UDP receiver on port 3333");
        if (sock >= 0) close(sock);
        vTaskDelete(NULL);
        return;
    }
    char payload[512];
    while (recvfrom(sock, payload, sizeof(payload), 0, NULL, NULL) >= 0) { }
    close(sock);
    vTaskDelete(NULL);
}

void wifi_sta_init(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_event_group = xEventGroupCreate();
    if (wifi_event_group == NULL) {
        ESP_LOGE(TAG, "Failed to create Wi-Fi event group");
        return;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    if (sta_netif == NULL) {
        ESP_LOGE(TAG, "Failed to create default STA netif");
        return;
    }

    // Static IP for RX1.
    // TX/AP is 192.168.4.1, RX1 is 192.168.4.2.
    esp_netif_dhcpc_stop(sta_netif);

    esp_netif_ip_info_t ip_info = {0};
    ip_info.ip.addr      = ESP_IP4TOADDR(RX_IP_A, RX_IP_B, RX_IP_C, RX_IP_D);
    ip_info.gw.addr      = ESP_IP4TOADDR(192, 168, 4, GW_IP_D);
    ip_info.netmask.addr = ESP_IP4TOADDR(255, 255, 255, 0);
    ESP_ERROR_CHECK(esp_netif_set_ip_info(sta_netif, &ip_info));

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    // CSI must be enabled both here and at build time with
    // CONFIG_ESP_WIFI_CSI_ENABLED=y.
    cfg.csi_enable = 1;

    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        &event_handler,
        NULL,
        NULL));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT,
        IP_EVENT_STA_GOT_IP,
        &event_handler,
        NULL,
        NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20));

    // Disable Wi-Fi power saving for stable CSI timing.
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "Waiting for connection...");

    xEventGroupWaitBits(
        wifi_event_group,
        WIFI_CONNECTED_BIT,
        pdFALSE,
        pdTRUE,
        portMAX_DELAY);
    configASSERT(xTaskCreate(udp_sink_task, "udp_sink", 3072, NULL, 4, NULL) == pdPASS);
}