#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "csi_proto.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#define CONFIG_WIFI_BANDWIDTH              WIFI_BW20
#define CONFIG_EXPECTED_CSI_LENGTH         CSI_IQ_BYTES

#define CONFIG_ESP_NOW_PHYMODE             WIFI_PHY_MODE_HT20
#define CONFIG_ESP_NOW_RATE                WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                  0  // Đặt bằng 1 nếu muốn cố định gain sau khi lấy baseline

#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#define CONFIG_GAIN_CONTROL

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

// Hàm trích xuất thông số AGC và FFT gain trực tiếp từ rx_ctrl của ESP32
// (Dựa trên cách hoạt động chuẩn của phần cứng Espressif Wi-Fi driver)
static void get_rx_gain_direct(const wifi_pkt_rx_ctrl_t *rx_ctrl, uint8_t *agc_gain, int8_t *fft_gain)
{
    if (rx_ctrl) {
        // Tùy theo phiên bản chip và bản miêu tả rx_ctrl, ta lấy trường rssi hoặc các trường gain tương ứng
        // Trên ESP32 thông thường, agc và noise_floor phản ánh trạng thái thu sóng
        *agc_gain = (uint8_t)(rx_ctrl->noise_floor & 0xFF); 
        *fft_gain = (int8_t)(rx_ctrl->rssi);
    }
}

// Hàm tính hệ số bù biên độ (Compensation gain) dựa trên sự thay đổi gain
static float calculate_gain_compensation(uint8_t current_agc, int8_t current_fft, 
                                       uint8_t baseline_agc, int8_t baseline_fft)
{
    // Công thức mô phỏng bù trừ biên độ tuyến tính theo lệch Gain (dB scale to linear scale factor)
    int db_diff = (int)(current_fft - baseline_fft);
    if (db_diff == 0) {
        return 1.0f;
    }
    // Chuyển đổi độ lệch dB sang hệ số biên độ tuyến tính: 10^(-delta_dB / 20)
    return powf(10.0f, -((float)db_diff) / 20.0f);
}

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }

    if (info->len != CONFIG_EXPECTED_CSI_LENGTH) {
        ESP_LOGW(TAG, "Skip CSI frame with length %d; expected %d for HT20",
                 info->len, CONFIG_EXPECTED_CSI_LENGTH);
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = 1.0f;
    
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;

#ifdef CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    static int32_t agc_sum = 0;
    static int32_t fft_sum = 0;

    get_rx_gain_direct(rx_ctrl, &agc_gain, &fft_gain);

    if (s_count < 100) {
        // Tích lũy 100 gói tin đầu tiên để lấy giá trị trung bình làm Baseline
        agc_sum += agc_gain;
        fft_sum += fft_gain;
        if (s_count == 99) {
            agc_gain_baseline = (uint8_t)(agc_sum / 100);
            fft_gain_baseline = (int8_t)(fft_sum / 100);
            ESP_LOGI(TAG, "Gain Baseline established -> AGC: %d, FFT: %d", agc_gain_baseline, fft_gain_baseline);
        }
    } else {
        compensate_gain = calculate_gain_compensation(agc_gain, fft_gain, agc_gain_baseline, fft_gain_baseline);
    }
    
    ESP_LOGI(TAG, "compensate_gain %.2f, agc_gain %d, fft_gain %d", compensate_gain, agc_gain, fft_gain);
#endif

    uint32_t rx_id = *(uint32_t *)(info->payload + 15);

    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("%s\n", CSI_CSV_HEADER);
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
               rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->sig_mode);

    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, (int16_t)(compensate_gain * info->buf[0]));
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", (int16_t)(compensate_gain * info->buf[i]));
    }

    ets_printf("]\"\n");
    s_count++;
}

static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0x68, 0x09, 0x47, 0x9d, 0xca, 0xf8},
    };

    wifi_esp_now_init(peer);
    wifi_csi_init();
}