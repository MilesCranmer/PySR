import os
import sys


def get_juliaregistrypref_envvarname():
    # We have this function so that we can define this string in one place, instead of needing
    # to repeat ourselves in multiple places.
    name = "JULIA_PKG_SERVER_REGISTRY_PREFERENCE"
    return name


def backup_juliaregistrypref():
    name = get_juliaregistrypref_envvarname()
    if name in os.environ:
        old_value = os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"]
        # If the user has set the JULIA_PKG_SERVER_REGISTRY_PREFERENCE environment variable, then
        # we don't overwrite it.
        pass
    else:
        # If the user has not set the JULIA_PKG_SERVER_REGISTRY_PREFERENCE environment variable, then
        # we set it temporarily, and unset it when we are done.
        old_value = None

        # We will set the JULIA_PKG_SERVER_REGISTRY_PREFERENCE to `eager`.
        # Note: in upstream Julia/Pkg, the default value is `conservative`.
        # Therefore, by setting it to `eager`, we are overriding the default.
        # Because we are deviating from the default behavior that would normally be expected in Julia,
        # I think that it's a good idea to print an informational message, so that the user is aware
        # of the change being made.
        info_msg = """\
            INFO: Attempting to use the `eager` registry flavor of the Julia General registry from the Julia Pkg server.
            If any errors are encountered, try setting the `{name}` environment variable to `conservative`.
        """.format(
            name=name  # TODO: get rid of this line?
        )
        print(info_msg, file=sys.stderr)
        os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = "eager"
    return old_value


def restore_juliaregistrypref(old_value):
    name = get_juliaregistrypref_envvarname()
    if old_value is None:
        # Delete the JULIA_PKG_SERVER_REGISTRY_PREFERENCE environment variable that we set:
        os.environ.pop(name)
    else:
        # Restore the original value of the JULIA_PKG_SERVER_REGISTRY_PREFERENCE environment variable:
        os.environ[name] = old_value
    return None


def with_juliaregistrypref(f, *args):
    name = get_juliaregistrypref_envvarname()
    old_value = backup_juliaregistrypref()
    try:
        f(*args)
    except:
        error_msg = """\
            ERROR: Encountered a network error.
            Are you behind a firewall, or are there network restrictions that would prevent access
            to certain websites or domains?
            Try setting the `{name}` environment variable to `conservative`.
        """.format(
            name=name  # TODO: get rid of this line?
        )
        if old_value is not None:
            print("", file=sys.stderr)
            print(error_msg, file=sys.stderr)
            print("", file=sys.stderr)

            # In this case, we know that the user had set JULIA_PKG_SERVER_REGISTRY_PREFERENCE,
            # and thus we respect that value, and we don't override it.
            # So, in this case, we just rethrow the caught exception:
            restore_juliaregistrypref(
                old_value
            )  # Restore the old value BEFORE we re-throw.
            raise  # Now, re-throw the caught exception.
        else:
            # In this case, the user had not set JULIA_PKG_SERVER_REGISTRY_PREFERENCE.
            # So we had initially tried with JULIA_PKG_SERVER_REGISTRY_PREFERENCE=eager, but that
            # resulted in an exception.
            # So let us now try with JULIA_PKG_SERVER_REGISTRY_PREFERENCE=conservative
            os.environ[name] = "conservative"
            try:
                # Note: after changing the value of `JULIA_PKG_SERVER_REGISTRY_PREFERENCE`,
                # you need to run `Pkg.Registry.update()` again. Otherwise the change will not take effect.
                # `juliacall` will automatically do this for us, so we don't need to do it ourselves.
                #
                # See line 334 here:
                # https://github.com/JuliaPy/pyjuliapkg/blob/3a2c66019f098c1ebf84f933a46e7ca70e82792b/src/juliapkg/deps.py#L334-L334
                f(args)
            except:
                print("", file=sys.stderr)
                print(error_msg, file=sys.stderr)
                print("", file=sys.stderr)
                # Now, we just rethrow the caught exception:
                restore_juliaregistrypref(
                    old_value
                )  # Restore the old value BEFORE we re-throw.
                raise  # Now, re-throw the caught exception.
    return None
