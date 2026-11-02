
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_now.h"

#include "csi_proto.h"

/* ================= CONFIGURATION ================= */

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#define CONFIG_WIFI_BANDWIDTH              WIFI_BW20

#define CONFIG_EXPECTED_CSI_LENGTH         CSI_IQ_BYTES

#define CONFIG_CSI_QUEUE_LENGTH            64
#define CONFIG_CSI_TASK_STACK_SIZE         4096
#define CONFIG_CSI_TASK_PRIORITY           5

#define CONFIG_UART_BAUDRATE               2000000

/*
 * MAC Address of ESP32 Sender.
 * Sender uses 1A:00:00:00:00:00.
 *
 * Receiver must use its own MAC address.
 */
static const uint8_t CONFIG_CSI_SEND_MAC[] = {
    0x1A, 0x00, 0x00, 0x00, 0x00, 0x00
};

static const char *TAG = "csi_recv";

/* ESP-NOW application payload layout.
 *
 * Existing Sender sends:
 *     uint32_t count
 *
 * Existing Receiver reads:
 *     info->payload + 15
 *
 * Verify this offset against the ESP-IDF version and
 * actual CSI payload layout used by the project.
 */
#define CONFIG_RX_ID_OFFSET 15

/* ================= DATA STRUCTURE ================= */

typedef struct {
    uint32_t rx_id;

    uint8_t mac[6];

    uint16_t csi_len;

    bool first_word_invalid;

    wifi_pkt_rx_ctrl_t rx_ctrl;

    int8_t csi_data[CSI_IQ_BYTES];

} csi_frame_t;

/* ================= GLOBAL OBJECTS ================= */

static QueueHandle_t csi_queue = NULL;

/*
 * Statistics:
 * These counters are updated by CSI callback or UART task.
 */
static volatile uint32_t s_callback_count = 0;
static volatile uint32_t s_queue_drop_count = 0;
static volatile uint32_t s_invalid_length_count = 0;
static volatile uint32_t s_invalid_payload_count = 0;

static uint32_t s_uart_count = 0;

/* ================= WIFI INITIALIZATION ================= */

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(esp_netif_init());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    ESP_ERROR_CHECK(
        esp_wifi_set_storage(WIFI_STORAGE_RAM)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_bandwidth(
            WIFI_IF_STA,
            CONFIG_WIFI_BANDWIDTH
        )
    );

    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(
        esp_wifi_set_ps(WIFI_PS_NONE)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_channel(
            CONFIG_LESS_INTERFERENCE_CHANNEL,
            WIFI_SECOND_CHAN_NONE
        )
    );

    /*
     * IMPORTANT:
     * Do not set Receiver MAC to Sender MAC.
     * Receiver keeps its own hardware MAC address.
     */
}

/* ================= ESP-NOW INITIALIZATION ================= */

static void wifi_esp_now_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());

    /*
     * No peer registration is required just to receive
     * unencrypted ESP-NOW packets.
     *
     * Sender broadcasts packets.
     */
}

/* ================= CSI CALLBACK ================= */

static void wifi_csi_rx_cb(
    void *ctx,
    wifi_csi_info_t *info
)
{
    (void)ctx;

    if (info == NULL || info->buf == NULL) {
        return;
    }

    /*
     * Only accept packets from our Sender.
     */
    if (memcmp(
            info->mac,
            CONFIG_CSI_SEND_MAC,
            sizeof(CONFIG_CSI_SEND_MAC)
        ) != 0) {
        return;
    }

    s_callback_count++;

    /*
     * Keep a fixed CSI length for compatibility
     * with the existing preprocessing pipeline.
     */
    if (info->len != CONFIG_EXPECTED_CSI_LENGTH) {
        s_invalid_length_count++;
        return;
    }

    /*
     * Validate application payload before reading rx_id.
     */
    if (info->payload == NULL ||
        info->payload_len <
            CONFIG_RX_ID_OFFSET + sizeof(uint32_t)) {

        s_invalid_payload_count++;
        return;
    }

    /*
     * Allocate frame locally.
     *
     * Copy all required data before callback returns.
     */
    csi_frame_t frame = {0};

    memcpy(
        frame.mac,
        info->mac,
        sizeof(frame.mac)
    );

    memcpy(
        &frame.rx_id,
        info->payload + CONFIG_RX_ID_OFFSET,
        sizeof(frame.rx_id)
    );

    frame.csi_len = info->len;

    frame.first_word_invalid = info->first_word_invalid;

    memcpy(
        &frame.rx_ctrl,
        &info->rx_ctrl,
        sizeof(frame.rx_ctrl)
    );

    memcpy(
        frame.csi_data,
        info->buf,
        info->len
    );

    /*
     * Non-blocking queue operation.
     *
     * Never wait for UART inside CSI callback.
     */
    if (xQueueSend(csi_queue, &frame, 0) != pdTRUE) {
        s_queue_drop_count++;
    }
}

/* ================= CSI INITIALIZATION ================= */

static void wifi_csi_init(void)
{
    ESP_ERROR_CHECK(
        esp_wifi_set_promiscuous(true)
    );

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };

    ESP_ERROR_CHECK(
        esp_wifi_set_csi_config(&csi_config)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_csi_rx_cb(
            wifi_csi_rx_cb,
            NULL
        )
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_csi(true)
    );
}

/* ================= CSI UART OUTPUT ================= */

/*
 * Format one CSI frame into a single output buffer.
 *
 * This function runs in UART task, NOT in CSI callback.
 */
static void print_csi_frame(const csi_frame_t *frame)
{
    char output[1024];

    const wifi_pkt_rx_ctrl_t *rx_ctrl =
        &frame->rx_ctrl;

    int pos = snprintf(
        output,
        sizeof(output),

        "CSI_DATA,%" PRIu32
        ",%02X:%02X:%02X:%02X:%02X:%02X"
        ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d"
        ",%d,%d,%d,%d,%d,%d,%d,%d,%d",

        frame->rx_id,

        frame->mac[0],
        frame->mac[1],
        frame->mac[2],
        frame->mac[3],
        frame->mac[4],
        frame->mac[5],

        rx_ctrl->rssi,
        rx_ctrl->rate,
        rx_ctrl->sig_mode,
        rx_ctrl->mcs,
        rx_ctrl->cwb,
        rx_ctrl->smoothing,
        rx_ctrl->not_sounding,
        rx_ctrl->aggregation,
        rx_ctrl->stbc,
        rx_ctrl->fec_coding,
        rx_ctrl->sgi,
        rx_ctrl->noise_floor,
        rx_ctrl->ampdu_cnt,
        rx_ctrl->channel,
        rx_ctrl->secondary_channel,
        rx_ctrl->timestamp,
        rx_ctrl->ant,
        rx_ctrl->sig_len,
        rx_ctrl->sig_mode
    );

    if (pos < 0 || pos >= sizeof(output)) {
        return;
    }

    int written = snprintf(
        output + pos,
        sizeof(output) - pos,
        ",%u,%d,\"[%d",

        frame->csi_len,
        frame->first_word_invalid,
        frame->csi_data[0]
    );

    if (written < 0 ||
        written >= sizeof(output) - pos) {
        return;
    }

    pos += written;

    /*
     * Append remaining CSI I/Q values.
     */
    for (int i = 1; i < frame->csi_len; i++) {

        written = snprintf(
            output + pos,
            sizeof(output) - pos,
            ",%d",
            frame->csi_data[i]
        );

        if (written < 0 ||
            written >= sizeof(output) - pos) {
            return;
        }

        pos += written;
    }

    /*
     * Finish CSV line.
     */
    written = snprintf(
        output + pos,
        sizeof(output) - pos,
        "]\"\n"
    );

    if (written < 0 ||
        written >= sizeof(output) - pos) {
        return;
    }

    /*
     * One output operation per CSI frame.
     */
    printf("%s", output);

    s_uart_count++;
}

/* ================= UART TASK ================= */

static void csi_uart_task(void *arg)
{
    (void)arg;

    csi_frame_t frame;

    uint32_t last_report_tick = xTaskGetTickCount();

    uint32_t last_callback_count = 0;
    uint32_t last_uart_count = 0;

    /*
     * Print CSV header once.
     */
    printf("%s\n", CSI_CSV_HEADER);

    while (1) {

        /*
         * Wait for CSI frame from queue.
         */
        if (xQueueReceive(
                csi_queue,
                &frame,
                pdMS_TO_TICKS(100)
            ) == pdTRUE) {

            print_csi_frame(&frame);
        }

        /*
         * Report statistics every 5 seconds.
         */
        uint32_t now = xTaskGetTickCount();

        if ((now - last_report_tick) >=
            pdMS_TO_TICKS(5000)) {

            uint32_t callback_count =
                s_callback_count;

            uint32_t uart_count =
                s_uart_count;

            ESP_LOGI(
                TAG,
                "CSI Stats | Callback: %" PRIu32
                " (+%" PRIu32 ")"
                " | UART: %" PRIu32
                " (+%" PRIu32 ")"
                " | Queue Drop: %" PRIu32
                " | Invalid Length: %" PRIu32
                " | Invalid Payload: %" PRIu32
                " | Queue Remaining: %u",

                callback_count,
                callback_count - last_callback_count,

                uart_count,
                uart_count - last_uart_count,

                s_queue_drop_count,
                s_invalid_length_count,
                s_invalid_payload_count,

                (unsigned int)uxQueueMessagesWaiting(csi_queue)
            );

            last_callback_count = callback_count;
            last_uart_count = uart_count;

            last_report_tick = now;
        }
    }
}

/* ================= APPLICATION ================= */

void app_main(void)
{
    /*
     * Initialize NVS.
     */
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {

        ESP_ERROR_CHECK(nvs_flash_erase());

        ret = nvs_flash_init();
    }

    ESP_ERROR_CHECK(ret);

    /*
     * Create queue before enabling CSI callback.
     */
    csi_queue = xQueueCreate(
        CONFIG_CSI_QUEUE_LENGTH,
        sizeof(csi_frame_t)
    );

    if (csi_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI queue");
        abort();
    }

    /*
     * Initialize Wi-Fi.
     */
    wifi_init();

    /*
     * Initialize ESP-NOW.
     */
    wifi_esp_now_init();

    /*
     * Create UART processing task.
     */
    BaseType_t task_result = xTaskCreate(
        csi_uart_task,
        "csi_uart_task",
        CONFIG_CSI_TASK_STACK_SIZE,
        NULL,
        CONFIG_CSI_TASK_PRIORITY,
        NULL
    );

    if (task_result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create CSI UART task");
        abort();
    }

    /*
     * Enable CSI after queue and task are ready.
     */
    wifi_csi_init();

    ESP_LOGI(TAG, "================ CSI RECV ================");

    ESP_LOGI(
        TAG,
        "Channel: %d | Expected CSI length: %d | Queue: %d",

        CONFIG_LESS_INTERFERENCE_CHANNEL,
        CONFIG_EXPECTED_CSI_LENGTH,
        CONFIG_CSI_QUEUE_LENGTH
    );
}
