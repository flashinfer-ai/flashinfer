// SPDX - FileCopyrightText : 2023-2035 FlashInfer team.
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#ifndef FLASHINFER_EXCEPTION_H_
#define FLASHINFER_EXCEPTION_H_

#include <exception>
#include <sstream>

namespace flashinfer
{

class Error : public std::exception
{
private:
    std::string message_;

public:
    Error(const std::string &func,
          const std::string &file,
          int line,
          const std::string &message)
    {
        std::ostringstream oss;
        oss << "Error in function '" << func << "' "
            << "at " << file << ":" << line << ": " << message;
        message_ = oss.str();
    }

    virtual const char *what() const noexcept override
    {
        return message_.c_str();
    }
};

#define FLASHINFER_ERROR(message)                                              \
    throw Error(__FUNCTION__, __FILE__, __LINE__, message)

#define FLASHINFER_CHECK(condition, message)                                   \
    if (!(condition)) {                                                        \
        FLASHINFER_ERROR(message);                                             \
    }

} // namespace flashinfer

#endif // FLASHINFER_EXCEPTION_H_
