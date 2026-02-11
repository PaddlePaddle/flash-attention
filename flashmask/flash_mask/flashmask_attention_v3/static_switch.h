/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 *
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#define FLASH_MASK_SWITCH(LTE_COND, UTS_COND, LTS_CONST_NAME, UTS_CONST_NAME, ...) \
  [&] {                                                                                                      \
    if (LTE_COND) {                                                                                          \
      constexpr static bool LTS_CONST_NAME = true;                                                           \
        if (UTS_COND) {                                                                                      \
          constexpr static bool UTS_CONST_NAME = true;                                                       \
          return __VA_ARGS__();                                                                              \
        } else {                                                                                             \
          constexpr static bool UTS_CONST_NAME = false;                                                      \
          return __VA_ARGS__();                                                                              \
        }                                                                                                    \
    } else {                                                                                                 \
      constexpr static bool LTS_CONST_NAME = false;                                                          \
        if (UTS_COND) {                                                                                      \
          constexpr static bool UTS_CONST_NAME = true;                                                       \
          return __VA_ARGS__();                                                                              \
        } else {                                                                                             \
          constexpr static bool UTS_CONST_NAME = false;                                                      \
          return __VA_ARGS__();                                                                              \
        }                                                                                                    \
    }                                                                                                        \
  }()

#ifdef FLASHMASK_V3_DISABLE_LOCAL
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      constexpr static bool LOCAL_CONST_NAME = false;                                            \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#else
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      } else if (LOCAL_COND) {                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = true;                                           \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#endif

#ifdef FLASHMASK_V3_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_PAGEDKV
  #define PAGEDKV_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PAGEDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_SPLIT
  #define SPLIT_SWITCH(COND, CONST_NAME, ...)                                                    \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SPLIT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_APPENDKV
  #define APPENDKV_SWITCH(COND, CONST_NAME, ...)                                                 \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define APPENDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_PACKGQA
  #define PACKGQA_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PACKGQA_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_VARLEN
  #define VARLEN_SWITCH(COND, CONST_NAME, ...)                                                   \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VARLEN_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_CLUSTER
  #define CLUSTER_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define CLUSTER_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHMASK_V3_DISABLE_SM8x
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    constexpr static int ARCH_NAME = 90;                                                         \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    if (ARCH == 86 || ARCH == 89) {                                                              \
      constexpr static int ARCH_NAME = 86;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (ARCH < 90) {                                                                      \
      constexpr static int ARCH_NAME = 80;                                                       \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int ARCH_NAME = 90;                                                       \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
#endif

#ifndef FLASHMASK_V3_ENABLE_VCOLMAJOR
  #define VCOLMAJOR_SWITCH(COND, CONST_NAME, ...)                                                \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VCOLMAJOR_SWITCH BOOL_SWITCH
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)                                                             \
  [&] {                                                                                          \
    if (HEADDIM == 64) {                                                                         \
      constexpr static int kHeadSize = 64;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 128) {                                                                 \
      constexpr static int kHeadSize = 128;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 256) {                                                                 \
      constexpr static int kHeadSize = 256;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
