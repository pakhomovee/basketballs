#include "mock/mock.h"

namespace mock {

std::string Mock::Get() const {
    return data_;
}

}  // namespace mock
