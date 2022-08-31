
#include <iostream>
#include <coroutine>
#include <concepts>
#include <memory>
#include <filesystem>
#include <complex>
#include <vector>
#include <list>
#include <numbers>
#include <bit>

#include <CL/sycl.hpp>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <linux/input-event-codes.h>
#include <wayland-client.h>


inline namespace wayland_client_helper
{
    template <class> constexpr std::nullptr_t wl_interface_ptr = nullptr;
#define INTERN_WL_INTERFACE(wlx)                                        \
    template <> constexpr wl_interface const *const wl_interface_ptr<wlx> = &wlx##_interface;
    INTERN_WL_INTERFACE(wl_display);
    INTERN_WL_INTERFACE(wl_registry);
    INTERN_WL_INTERFACE(wl_compositor);
    INTERN_WL_INTERFACE(wl_shell);
    INTERN_WL_INTERFACE(wl_seat);
    INTERN_WL_INTERFACE(wl_keyboard);
    INTERN_WL_INTERFACE(wl_pointer);
    INTERN_WL_INTERFACE(wl_touch);
    INTERN_WL_INTERFACE(wl_shm);
    INTERN_WL_INTERFACE(wl_surface);
    INTERN_WL_INTERFACE(wl_shell_surface);
    INTERN_WL_INTERFACE(wl_buffer);
    INTERN_WL_INTERFACE(wl_shm_pool);
    INTERN_WL_INTERFACE(wl_callback);
    INTERN_WL_INTERFACE(wl_output);
#undef INTERN_WL_INTERFACE
    template <class T>
    concept wl_client_t = std::same_as<decltype (wl_interface_ptr<T>), wl_interface const *const>;
    template <wl_client_t T, class Ch>
    auto& operator << (std::basic_ostream<Ch>& output, T const* ptr) noexcept {
        return output << static_cast<void const*>(ptr)
                      << '['
                      << wl_interface_ptr<T>->name
                      << ']';
    }
    template <wl_client_t T>
    [[nodiscard]] auto attach_unique(T* ptr) noexcept {
        static constexpr auto deleter = [](T* ptr) noexcept -> void {
            std::cout << ptr << " deleting." << std::endl;
            static constexpr auto interface_addr = wl_interface_ptr<T>;
            if      constexpr (interface_addr == std::addressof(wl_display_interface)) {
                wl_display_disconnect(ptr);
            }
            else if constexpr (interface_addr == std::addressof(wl_keyboard_interface)) {
                wl_keyboard_release(ptr);
            }
            else if constexpr (interface_addr == std::addressof(wl_pointer_interface)) {
                wl_pointer_release(ptr);
            }
            else if constexpr (interface_addr == std::addressof(wl_touch_interface)) {
                wl_touch_release(ptr);
            }
            else {
                wl_proxy_destroy(reinterpret_cast<wl_proxy*>(ptr));
            }
        };
        return std::unique_ptr<T, decltype (deleter)>(ptr, deleter);
    }
    template <wl_client_t T>
    using unique_ptr_t = decltype (attach_unique(std::declval<T*>()));
    /////////////////////////////////////////////////////////////////////////////
    [[nodiscard]]
    inline auto create_shm_buffer(wl_shm* shm, size_t cx, size_t cy, uint32_t** pixels) noexcept
    -> wl_buffer*
    {
        // Check the environment
        std::string_view xdg_runtime_dir = std::getenv("XDG_RUNTIME_DIR");
        if (xdg_runtime_dir.empty() || !std::filesystem::exists(xdg_runtime_dir)) {
            std::cerr << "This program requires XDG_RUNTIME_DIR setting..." << std::endl;
            return nullptr;
        }
        std::string_view tmp_file_title = "/weston-shared-XXXXXX";
        if (1024 <= xdg_runtime_dir.size() + tmp_file_title.size()) {
            std::cerr << "The path of XDG_RUNTIME_DIR is too long..." << std::endl;
            return nullptr;
        }
        char tmp_path[1024] = { };
        auto p = std::strcat(tmp_path, xdg_runtime_dir.data());
        std::strcat(p, tmp_file_title.data());
        int fd = mkostemp(tmp_path, O_CLOEXEC);
        if (fd >= 0) {
            unlink(tmp_path);
        }
        else {
            std::cerr << "mkostemp failed..." << std::endl;
            return nullptr;
        }
        if (ftruncate(fd, 4*cx*cy) < 0) {
            std::cerr << "ftruncate failed..." << std::endl;
            close(fd);
            return nullptr;
        }
        auto data = mmap(nullptr, 4*cx*cy, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            std::cerr << "mmap failed..." << std::endl;
            close(fd);
            return nullptr;
        }
        *pixels = reinterpret_cast<uint32_t*>(data);
        return wl_shm_pool_create_buffer(attach_unique(wl_shm_create_pool(shm, fd, 4*cx*cy)).get(),
                                         0,
                                         cx, cy,
                                         cx * 4,
                                         WL_SHM_FORMAT_XRGB8888);
    }
}

void windowing(wl_display* display, wl_registry* registry, auto const& globals) noexcept;

void rendering(sycl::queue* const que,
               uint32_t* const pixels,
               sycl::range<2> dim,
               std::vector<std::complex<int>> const& vertices) noexcept;

int main() {
    if (auto display = attach_unique(wl_display_connect(nullptr))) {
        if (auto registry = attach_unique(wl_display_get_registry(display.get()))) {
            try {
                std::list<std::tuple<uint32_t, std::string, uint32_t>> globals;
                wl_registry_listener listener {
                    .global = [](void* data, auto, auto... rest) {
                        auto globals_ptr = reinterpret_cast<decltype (globals)*>(data);
                        globals_ptr->push_back(std::tuple(rest...));
                    },
                    .global_remove = [](void* data, auto, uint32_t name) {
                        auto globals_ptr = reinterpret_cast<decltype (globals)*>(data);
                        std::remove_if(globals_ptr->begin(),
                                       globals_ptr->end(),
                                       [&](auto const& item) noexcept -> bool {
                                           return std::get<0>(item) == name;
                                       });
                    },
                };
                wl_registry_add_listener(registry.get(), &listener, &globals);
                wl_display_roundtrip(display.get());
                windowing(display.get(), registry.get(), globals);
            }
            catch (std::exception& ex) {
                std::cerr << "Exception: " << ex.what() << std::endl;
            }
        }
    }
    return 0;
}

void windowing(wl_display* display, wl_registry* registry, auto const& globals) noexcept {
    constexpr int32_t cx = 1024;
    constexpr int32_t cy =  768;
    wl_compositor* compositor = nullptr;
    wl_shell* shell = nullptr;
    wl_seat* seat = nullptr;
    wl_shm* shm = nullptr;
    for (auto const& item : globals) {
        auto const& [name, interface, version] = item;
        if (interface == wl_compositor_interface.name) {
            compositor = reinterpret_cast<wl_compositor*>(wl_registry_bind(registry,
                                                                           name,
                                                                           &wl_compositor_interface,
                                                                           version));
        }
        else if (interface == wl_shell_interface.name) {
            shell = reinterpret_cast<wl_shell*>(wl_registry_bind(registry,
                                                                 name,
                                                                 &wl_shell_interface,
                                                                 version));
        }
        else if (interface == wl_seat_interface.name) {
            seat = reinterpret_cast<wl_seat*>(wl_registry_bind(registry,
                                                               name,
                                                               &wl_seat_interface,
                                                               version));
        }
        else if (interface == wl_shm_interface.name) {
            shm = reinterpret_cast<wl_shm*>(wl_registry_bind(registry,
                                                             name,
                                                             &wl_shm_interface,
                                                             version));
        }
    }
    if (!compositor || !shell || !seat || !shm) {
        std::cerr << "Some required globals are missing..." << std::endl;
        return ;
    }
    auto compositor_ptr = attach_unique(compositor);
    auto shell_ptr = attach_unique(shell);
    auto seat_ptr = attach_unique(seat);
    auto shm_ptr = attach_unique(shm);
    std::vector<uint32_t> formats;
    wl_shm_listener shm_listener {
        .format = [](void* data, auto, uint32_t format) noexcept {
            reinterpret_cast<decltype (formats)*>(data)->push_back(format);
        },
    };
    if (wl_shm_add_listener(shm, &shm_listener, &formats)) {
        std::cerr << "wl_shm_add_listener failed..." << std::endl;
        return ;
    }
    uint32_t seat_capability = 0;
    wl_seat_listener seat_listener {
        .capabilities = [](void* data, auto, uint32_t caps) noexcept {
            *reinterpret_cast<uint32_t*>(data) = caps;
        },
        .name = [](auto, auto, char const* name) noexcept {
        }
    };
    if (wl_seat_add_listener(seat, &seat_listener, &seat_capability)) {
        std::cerr << "wl_seat_add_listener failed..." << std::endl;
        return ;
    }
    wl_display_roundtrip(display);
    if (std::none_of(formats.begin(), formats.end(), [](auto item) noexcept {
        return item == WL_SHM_FORMAT_ARGB8888;
    }))
    {
        std::cerr << "WL_SHM_FORMAT_ARGB8888 required..." << std::endl;
        return ;
    }
    if (!(seat_capability & WL_SEAT_CAPABILITY_POINTER) ||
        !(seat_capability & WL_SEAT_CAPABILITY_KEYBOARD))
    {
        std::cerr << "Keyboad and pointers required..." << std::endl;
        return ;
    }
    auto keyboard = wl_seat_get_keyboard(seat);
    if (!keyboard) {
        std::cerr << "wl_seat_get_keyboard failed..." << std::endl;
        return ;
    }
    auto keyboard_ptr = attach_unique(keyboard);
    bool quit = false;
    wl_keyboard_listener keyboard_listener {
        .keymap = [](auto...) noexcept { },
        .enter = [](auto...) noexcept { },
        .leave = [](auto...) noexcept { },
        .key = [](auto data, auto, auto serial, auto time, auto key, auto state) noexcept {
            if (state == 0 && (key == 1 || key == 16)) {
                *reinterpret_cast<bool*>(data) = true;
            }
        },
        .modifiers = [](auto...) noexcept { },
        .repeat_info = [](auto...) noexcept { },
    };
    if (wl_keyboard_add_listener(keyboard, &keyboard_listener, &quit)) {
        std::cerr << "wl_keyboard_add_listener failed..." << std::endl;
        return ;
    }
    auto pointer = wl_seat_get_pointer(seat);
    if (!pointer) {
        std::cerr << "wl_seat_get_pointer failed..." << std::endl;
        return ;
    }
    auto pointer_ptr = attach_unique(pointer);
    std::vector<std::complex<int>> vertices{{}};
    assert(vertices.empty() == false);
    wl_pointer_listener pointer_listener {
        .enter = [](auto...) noexcept { },
        .leave = [](auto...) noexcept { },
        .motion = [](auto data, auto, auto time, auto x, auto y) noexcept {
            auto vertices = reinterpret_cast<std::vector<std::complex<int>>*>(data);
            auto& cursor = vertices->back();
            cursor = { wl_fixed_to_double(x), wl_fixed_to_double(y) };
        },
        .button = [](auto data, auto, auto, auto, auto button, auto state) /*noexcept*/ {
            if (button == BTN_RIGHT && state) {
                auto vertices = reinterpret_cast<std::vector<std::complex<int>>*>(data);
                vertices->push_back(vertices->back());
            }
        },
        .axis = [](auto...) noexcept { },
        .frame = [](auto...) noexcept { },
        .axis_source = [](auto...) noexcept { },
        .axis_stop = [](auto...) noexcept { },
        .axis_discrete = [](auto...) noexcept { },
    };
    if (wl_pointer_add_listener(pointer, &pointer_listener, &vertices)) {
        std::cerr << "wl_pointer_add_listener failed..." << std::endl;
        return ;
    }
    auto surface = wl_compositor_create_surface(compositor);
    if (!surface) {
        std::cerr << "wl_compositor_create_surface failed..." << std::endl;
        return ;
    }
    auto surface_ptr = attach_unique(surface);
    auto shell_surface = wl_shell_get_shell_surface(shell, surface);
    if (!shell_surface) {
        std::cerr << wl_shell_get_shell_surface(shell, surface) << std::endl;
        return ;
    }
    auto shell_surface_ptr = attach_unique(shell_surface);
    wl_shell_surface_listener shellsurf_listener {
        .ping = [](auto, auto shellsurf, auto serial) noexcept {
            wl_shell_surface_pong(shellsurf, serial);
            std::cout << "Pinged and ponged." << std::endl;
        },
        .configure = [](auto...) noexcept {
            std::cout << "Configuring... (not support yet)" << std::endl;
        },
        .popup_done = [](auto...) noexcept {
            std::cerr << "Popup done." << std::endl;
        },
    };
    if (wl_shell_surface_add_listener(shell_surface, &shellsurf_listener, nullptr)) {
        std::cerr << "wl_shell_surface_add_listener failed..." << std::endl;
        return ;
    }
    uint32_t* pixels = nullptr;
    auto buffer = create_shm_buffer(shm, cx, cy, &pixels);
    if (!buffer) {
        std::cerr << "create_shm_buffer failed..." << std::endl;
        return ;
    }
    auto buffer_ptr = attach_unique(buffer);
    wl_shell_surface_set_toplevel(shell_surface);
    sycl::queue que;
    do {
        if (quit) break;
        rendering(&que, pixels, {cy, cx}, vertices);
        wl_surface_damage(surface, 0, 0, cx, cy);
        wl_surface_attach(surface, buffer, 0, 0);
        wl_surface_commit(surface);
        wl_display_flush(display);
    } while (wl_display_dispatch(display) != -1);
}

struct color {
    uint32_t argb;
    constexpr color(uint32_t argb) noexcept
        : argb(argb)
    {
    }
    constexpr color(std::byte x) noexcept
        : argb((uint32_t)x << 24 | (uint32_t)x << 16 |(uint32_t)x << 8 | (uint32_t)x)
    {
    }
    constexpr operator uint32_t() const noexcept { return this->argb; }
    uint32_t a() const noexcept {
        return (uint32_t) *(reinterpret_cast<std::byte const*>(this) + 3);
    }
    uint32_t r() const noexcept {
        return (uint32_t) *(reinterpret_cast<std::byte const*>(this) + 2);
    }
    uint32_t g() const noexcept {
        return (uint32_t) *(reinterpret_cast<std::byte const*>(this) + 1);
    }
    uint32_t b() const noexcept {
        return (uint32_t) *(reinterpret_cast<std::byte const*>(this) + 0);
    }
    color& operator*=(color src) noexcept {
        auto& dst = *this;
        auto q = (255 - src.a());
        auto a = std::min(255u, src.a() + ((dst.a() * q) / 255));
        auto r = std::min(255u, src.r() + ((dst.r() * q) / 255));
        auto g = std::min(255u, src.g() + ((dst.g() * q) / 255));
        auto b = std::min(255u, src.b() + ((dst.b() * q) / 255));
        this->argb = a << 24 | r << 16 | g << 8 | b;
        return *this;
    }
};

void rendering(sycl::queue* const que,
               uint32_t* const pixels,
               sycl::range<2> dim,
               std::vector<std::complex<int>> const& vertices) noexcept
{
    auto pix = sycl::buffer{pixels, dim};
    auto pts = sycl::buffer<std::complex<int>>{&vertices.front(), vertices.size()};
    que->submit([&](sycl::handler& h) noexcept {
        auto a = pix.get_access<sycl::access::mode::write>(h);
        h.parallel_for(dim, [=](auto idx) noexcept {
            a[idx] = 0xcc000000;
        });
    });
    que->submit([&](sycl::handler& h) noexcept {
        constexpr float phi = std::numbers::phi_v<float>;
        constexpr float tau = std::numbers::pi_v<float> * 2.0f;
        size_t P = 16;
        size_t M = (1 << P);
        auto apx = pix.get_access<sycl::access::mode::read_write>(h);
        auto apt = pts.get_access<sycl::access::mode::read>(h);
        h.parallel_for(vertices.size() << P, [=](auto idx) noexcept {
            int n = idx >> P;
            int m = idx & (M - 1);
            std::complex<float> pt = {
                static_cast<float>(apt[n].real()),
                static_cast<float>(apt[n].imag()),
            };
            pt += std::polar<float>(std::sqrt(m)/1.0f, m*tau*phi);
            color dst = apx[{(unsigned) pt.imag(), (unsigned) pt.real()}];
            dst *= color((std::byte) (255 - m * 255 / (M-1)));
            apx[{(unsigned) pt.imag(), (unsigned) pt.real()}] = dst;
        });
    });
}
