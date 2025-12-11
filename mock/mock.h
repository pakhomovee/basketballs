#include <string>

namespace mock {

class Mock {
   private:
    std::string data_;

   public:
    explicit Mock(std::string data) : data_(data) {}
    std::string Get() const;
};

}  // namespace mock
