/*
 * csi_proto.h - Giao thuc chung cho 2 thiet bi ESP32 cua do an CSI_HAR
 *
 *   csi_send (TX) : phat ESPNOW broadcast 32 byte @ CONFIG_CSI_SEND_RATE_HZ
 *   csi_recv (RX) : bat CSI 64 sub-carrier HT20, ghi CSV ra UART
 *
 * File nay la "nguon su that" duy nhat cho:
 *   1) cau truc payload cua goi tin ESPNOW
 *   2) bang ten cot CSV (giai quyet F4 + F6 cua Project-Plan.md)
 *
 * Moi thay doi o day deu phai dong bo ca 2 project (csi_send + csi_recv)
 * va tools/record.py.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_PROTO_VERSION 1

/* 'C','S','I','T' khi doc nho endian cua ESP32 (little-endian) */
#define CSI_TX_MAGIC       0x54495343u
#define CSI_TX_PAYLOAD_LEN 32

/* goi tin TX: 20 byte thong tin + 12 byte duoi chuyen 0 de giu do dai co dinh 32 */
typedef struct __attribute__((packed)) {
    uint32_t magic;    /* CSI_TX_MAGIC */
    uint32_t version;  /* CSI_PROTO_VERSION */
    uint32_t seq;      /* so thu tu goi tin, tang 1 moi goi, tran o 2^32 (~71.6 phut) */
    uint32_t t_us;     /* esp_timer_get_time() tai thiet bi TX, don vi micro giay */
    uint16_t flags;    /* bit0 = goi tin co payload hop le (gia tri 1 khi TX tao) */
    uint16_t reserved;
} csi_tx_payload_t;

/* HT20 layout used by this project: 64 interleaved complex subcarriers. */
#define CSI_SUBCARRIERS 64
/* moi sub-carrier 2 byte: I la byte chan, Q la byte le -> 64 * 2 = 128 byte */
#define CSI_IQ_BYTES     (CSI_SUBCARRIERS * 2)

/*
 * Bang ten cot CSV chuan cua thiet bi RX.
 * Thu tu nay phai khop chinh xac voi phan ghi trong csi_recv.c
 * va voi tools/record.py.
 *
 * Luu y ve ten cot (sua F4/F6):
 *   - "local_timestamp" la rx_ctrl->timestamp (don vi micro giay).
 */
#define CSI_CSV_HEADER \
    "type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding," \
    "aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel," \
    "secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data"

#define CSI_CSV_COLUMNS 25

/* Gia tri cot "type" */
#define CSI_CSV_TYPE "CSI_DATA"

#ifdef __cplusplus
}
#endif