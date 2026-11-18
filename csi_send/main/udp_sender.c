#include "udp_sender.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define UDP_PORT            3333
#define PACKET_SIZE         256
#define SEND_INTERVAL_MS    40

static const char *TAG = "UDP_TX";

static void udp_sender_task(void *pvParameters)
{
    char packet[PACKET_SIZE];
    memset(packet, 'A', sizeof(packet));

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Cannot create socket");
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in dest_addr = {0};
    dest_addr.sin_addr.s_addr = inet_addr("192.168.4.2");
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_PORT);
    ESP_LOGI(TAG, "Single STA: 192.168.4.2:%d, interval=%d ms", UDP_PORT, SEND_INTERVAL_MS);
    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        if (sendto(sock, packet, sizeof(packet), 0,
                   (struct sockaddr *)&dest_addr, sizeof(dest_addr)) < 0) {
            ESP_LOGW(TAG, "UDP send failed");
        }
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SEND_INTERVAL_MS));
    }
}

void udp_sender_start(void)
{
    xTaskCreate(udp_sender_task, "udp_sender_task", 4096, NULL, 5, NULL);
}