#pragma once
#include <nccl_device.h>
struct EqWindow { unsigned long long base, bytes; ncclWindow_t window; };
struct EqPending { int peer, valid; unsigned long long sequence; };
struct EqState { ncclDevComm comm; int count; EqWindow windows[64]; unsigned long long shadow_keys[4096], shadow_values[4096]; EqPending pending[2560]; };
